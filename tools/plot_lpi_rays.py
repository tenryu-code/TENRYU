#!/usr/bin/env python3
"""Plot CBET ray exchange and hot-electron paths from a TENRYU run."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover - dependency guard
    h5py = None  # type: ignore[assignment]

try:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LogNorm, SymLogNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Wedge
except ModuleNotFoundError:  # pragma: no cover - dependency guard
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
    pe = None  # type: ignore[assignment]
    LineCollection = None  # type: ignore[assignment]
    LogNorm = None  # type: ignore[assignment]
    SymLogNorm = None  # type: ignore[assignment]
    Line2D = None  # type: ignore[assignment]
    Patch = None  # type: ignore[assignment]
    Wedge = None  # type: ignore[assignment]

try:
    # Reuse the authoritative snapshot, laser-mesh, and trajectory loaders.
    from plot_laser_rays import (
        LaserMeshData,
        RayTrajectoryData,
        SnapshotFormatError,
        _collect_snapshots_from_dir,
        _read_mesh,
        _read_trajectory,
        _subsample_indices,
    )
except ImportError:  # pragma: no cover - module execution from repository root
    from tools.plot_laser_rays import (
        LaserMeshData,
        RayTrajectoryData,
        SnapshotFormatError,
        _collect_snapshots_from_dir,
        _read_mesh,
        _read_trajectory,
        _subsample_indices,
    )


CM_TO_UM = 1.0e4
ERG_PER_S_TO_TW = 1.0e-19
PROTON_MASS_G = 1.6726219e-24
ELECTRON_MASS_G = 9.1094e-28
ELEMENTARY_CHARGE_ESU = 4.8032e-10
C_LIGHT_CM_S = 2.99792458e10

HYDRO_RHO = "hydro/rho"
HYDRO_ZBAR = "hydro/zbar"
HYDRO_VOLFRAC = "hydro/volFrac"
HYDRO_CBET_GROSS = "hydro/cbet_gross_exchange"
HYDRO_HOT_E_Q = "hydro/hot_e_Q"
MESH_X_R = "mesh/x_r"
PORT_CAPTURE_PCROSS = "laser/port_section/port_capture_Pcross"
TRAJECTORY_GROUP = "laser/rays/trajectory"

FONT_FAMILY = [
    "Hiragino Sans",
    "Hiragino Kaku Gothic ProN",
    "sans-serif",
]


@dataclass(frozen=True)
class SnapshotRef:
    path: Path
    time_s: float


@dataclass
class RadialPlasma:
    edges_cm: np.ndarray
    centers_cm: np.ndarray
    rho_g_cc: np.ndarray
    zbar: np.ndarray
    n_hat: np.ndarray


@dataclass
class RayPath:
    index: int
    points_um: np.ndarray
    radius_cm: np.ndarray
    s_cm: np.ndarray
    power_erg_s: np.ndarray
    dlnp_ds_cm: np.ndarray


@dataclass
class ChannelSurface:
    index: int
    mechanism: str
    radius_cm: float
    angles_deg: np.ndarray


class FigureDataError(RuntimeError):
    """Raised when a requested figure cannot be produced from a snapshot."""


def _ensure_dependencies() -> None:
    if h5py is None:
        raise RuntimeError("h5py is required. Install dependency: pip install h5py")
    if plt is None:
        raise RuntimeError("matplotlib is required. Install dependency: pip install matplotlib")


def _warn(message: str) -> None:
    print(f"[warn] {message}", file=sys.stderr)


def _require_array(handle: h5py.File, path: str) -> np.ndarray:
    obj = handle.get(path)
    if obj is None:
        raise FigureDataError(f"missing required dataset '/{path}'")
    if not isinstance(obj, h5py.Dataset):
        raise FigureDataError(f"expected dataset at '/{path}'")
    return np.asarray(obj[...])


def _read_time(handle: h5py.File) -> float:
    for key in ("time", "t"):
        if key in handle.attrs:
            raw = np.asarray(handle.attrs[key], dtype=float)
            if raw.shape == () and np.isfinite(raw.item()):
                return float(raw.item())
    obj = handle.get("time_state/t")
    if isinstance(obj, h5py.Dataset):
        raw = np.asarray(obj[...], dtype=float).reshape(-1)
        if raw.size == 1 and np.isfinite(raw[0]):
            return float(raw[0])
    raise SnapshotFormatError(
        "missing scalar snapshot time in root attribute 'time'/'t' or '/time_state/t'"
    )


def _discover_snapshots(run_dir: Path) -> list[SnapshotRef]:
    paths = _collect_snapshots_from_dir(run_dir)
    snapshots: list[SnapshotRef] = []
    for path in paths:
        try:
            with h5py.File(path, "r") as handle:
                time_s = _read_time(handle)
        except (OSError, SnapshotFormatError) as exc:
            _warn(f"omitting unreadable snapshot {path}: {exc}")
            continue
        snapshots.append(SnapshotRef(path=path, time_s=time_s))
    snapshots.sort(key=lambda item: (item.time_s, item.path.name))
    if not snapshots:
        raise FileNotFoundError(f"no readable indexed snapshots found under {run_dir}")
    return snapshots


def _load_frozen_config(run_dir: Path) -> tuple[dict[str, object] | None, Path | None]:
    paths = sorted((run_dir / "config").glob("*_frozen.json"))
    if not paths:
        _warn(f"no frozen config found under {run_dir / 'config'}")
        return None, None
    if len(paths) > 1:
        _warn(f"multiple frozen configs found; using deterministic first path {paths[0]}")
    try:
        value = json.loads(paths[0].read_text())
    except (OSError, ValueError) as exc:
        _warn(f"could not read frozen config {paths[0]}: {exc}")
        return None, paths[0]
    if not isinstance(value, dict):
        _warn(f"frozen config root is not an object: {paths[0]}")
        return None, paths[0]
    return value, paths[0]


def _parse_times_ns(value: str) -> list[float]:
    result: list[float] = []
    for raw_token in value.split(","):
        token = raw_token.strip().lower()
        if token.endswith("ns"):
            token = token[:-2].strip()
        if not token:
            raise ValueError("--times must be a comma-separated list in ns")
        time_ns = float(token)
        if not np.isfinite(time_ns) or time_ns < 0.0:
            raise ValueError("--times values must be finite and >= 0")
        result.append(time_ns)
    if not result:
        raise ValueError("--times must contain at least one time")
    return result


def _select_snapshots(
    snapshots: list[SnapshotRef],
    requested_times_ns: list[float] | None,
) -> list[SnapshotRef]:
    times_s = np.asarray([snapshot.time_s for snapshot in snapshots])
    if requested_times_ns is None:
        last_time_s = float(times_s[-1])
        targets_s = [0.50 * last_time_s, 0.95 * last_time_s]
    else:
        targets_s = [time_ns * 1.0e-9 for time_ns in requested_times_ns]

    selected: list[SnapshotRef] = []
    seen: set[Path] = set()
    for target_s in targets_s:
        index = int(np.argmin(np.abs(times_s - target_s)))
        snapshot = snapshots[index]
        if snapshot.path not in seen:
            selected.append(snapshot)
            seen.add(snapshot.path)
    return selected


def _time_tag(time_s: float) -> str:
    return f"{time_s * 1.0e9:.3f}"


def _atomic_mass_profile(
    handle: h5py.File,
    config: dict[str, object],
    n_cells: int,
) -> np.ndarray:
    materials_block = config.get("materials")
    materials = (
        materials_block.get("materials")
        if isinstance(materials_block, dict)
        else None
    )
    if not isinstance(materials, list) or not materials:
        raise FigureDataError(
            "missing frozen config field 'materials.materials[].A'"
        )

    atomic_masses: list[float] = []
    for material in materials:
        atomic_mass = material.get("A") if isinstance(material, dict) else None
        if (
            not isinstance(atomic_mass, (int, float))
            or not np.isfinite(atomic_mass)
            or atomic_mass <= 0.0
        ):
            raise FigureDataError(
                "invalid frozen config field 'materials.materials[].A'"
            )
        atomic_masses.append(float(atomic_mass))

    if len(atomic_masses) == 1:
        return np.full(n_cells, atomic_masses[0], dtype=float)

    volfrac = np.asarray(
        _require_array(handle, HYDRO_VOLFRAC),
        dtype=float,
    )
    if volfrac.shape != (n_cells, len(atomic_masses)):
        raise FigureDataError(
            "dataset '/hydro/volFrac' must have shape "
            f"({n_cells}, {len(atomic_masses)}), got {volfrac.shape}"
        )
    fractions = np.maximum(volfrac, 0.0)
    fraction_sum = np.sum(fractions, axis=1)
    inverse_a = np.sum(
        fractions / np.asarray(atomic_masses)[np.newaxis, :],
        axis=1,
    )
    valid = fraction_sum > 1.0e-30
    inverse_a[valid] /= fraction_sum[valid]
    result = np.full(n_cells, np.nan)
    valid &= np.isfinite(inverse_a) & (inverse_a > 1.0e-30)
    result[valid] = 1.0 / inverse_a[valid]
    return result


def _critical_density_cm3(config: dict[str, object]) -> float:
    laser = config.get("laser")
    wavelength_nm = laser.get("wavelength_nm") if isinstance(laser, dict) else None
    if (
        not isinstance(wavelength_nm, (int, float))
        or not np.isfinite(wavelength_nm)
        or wavelength_nm <= 0.0
    ):
        raise FigureDataError(
            "missing positive frozen config field 'laser.wavelength_nm'"
        )
    wavelength_cm = float(wavelength_nm) * 1.0e-7
    omega = 2.0 * math.pi * C_LIGHT_CM_S / wavelength_cm
    return (
        ELECTRON_MASS_G
        * omega
        * omega
        / (4.0 * math.pi * ELEMENTARY_CHARGE_ESU**2)
    )


def _read_radial_plasma(
    handle: h5py.File,
    config: dict[str, object],
) -> RadialPlasma:
    rho = np.asarray(_require_array(handle, HYDRO_RHO), dtype=float).reshape(-1)
    zbar = np.asarray(_require_array(handle, HYDRO_ZBAR), dtype=float).reshape(-1)
    edges = np.asarray(_require_array(handle, MESH_X_R), dtype=float).reshape(-1)
    if rho.size == 0 or zbar.size != rho.size:
        raise FigureDataError(
            "datasets '/hydro/rho' and '/hydro/zbar' must be compatible nonempty profiles"
        )
    if (
        edges.size != rho.size + 1
        or not np.all(np.isfinite(edges))
        or not np.all(np.diff(edges) > 0.0)
    ):
        raise FigureDataError(
            "dataset '/mesh/x_r' must contain n_cells+1 finite increasing edges"
        )
    a_eff = _atomic_mass_profile(handle, config, rho.size)
    n_crit = _critical_density_cm3(config)
    n_hat = (
        rho
        * np.maximum(zbar, 0.0)
        / (a_eff * PROTON_MASS_G * n_crit)
    )
    return RadialPlasma(
        edges_cm=edges,
        centers_cm=0.5 * (edges[:-1] + edges[1:]),
        rho_g_cc=rho,
        zbar=zbar,
        n_hat=n_hat,
    )


def _read_hydro_electron_density(
    handle: h5py.File,
    config: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    rho = np.asarray(_require_array(handle, HYDRO_RHO), dtype=float).reshape(-1)
    zbar = np.asarray(_require_array(handle, HYDRO_ZBAR), dtype=float).reshape(-1)
    edges = np.asarray(_require_array(handle, MESH_X_R), dtype=float).reshape(-1)
    if rho.size == 0 or zbar.size != rho.size:
        raise FigureDataError(
            "datasets '/hydro/rho' and '/hydro/zbar' must be compatible nonempty profiles"
        )
    if (
        edges.size != rho.size + 1
        or not np.all(np.isfinite(edges))
        or not np.all(np.diff(edges) > 0.0)
    ):
        raise FigureDataError(
            "dataset '/mesh/x_r' must contain n_cells+1 finite increasing edges"
        )
    # Material averaging follows material_atomic_mass_profile() in
    # tools/plot_multibeam_summary.py.
    atomic_mass = _atomic_mass_profile(handle, config, rho.size)
    electron_density = rho * zbar / (atomic_mass * PROTON_MASS_G)
    return 0.5 * (edges[:-1] + edges[1:]), electron_density


def _outermost_crossing_radius(
    centers_cm: np.ndarray,
    values: np.ndarray,
    target: float,
) -> float | None:
    for outer in range(values.size - 1, 0, -1):
        inner = outer - 1
        y_inner = float(values[inner])
        y_outer = float(values[outer])
        if not np.isfinite(y_inner) or not np.isfinite(y_outer):
            continue
        if y_outer == target:
            return float(centers_cm[outer])
        if y_inner == target:
            return float(centers_cm[inner])
        if (y_inner - target) * (y_outer - target) < 0.0:
            fraction = (target - y_inner) / (y_outer - y_inner)
            return float(
                centers_cm[inner]
                + fraction * (centers_cm[outer] - centers_cm[inner])
            )
    return None


def _cbet_band(edges_cm: np.ndarray, gross: np.ndarray) -> tuple[float, float]:
    if gross.size != edges_cm.size - 1:
        raise FigureDataError(
            "datasets '/hydro/cbet_gross_exchange' and '/mesh/x_r' have incompatible sizes"
        )
    finite = gross[np.isfinite(gross)]
    maximum = float(np.max(finite)) if finite.size else 0.0
    if maximum <= 0.0:
        raise FigureDataError(
            "dataset '/hydro/cbet_gross_exchange' has no positive exchange"
        )
    indices = np.flatnonzero(np.isfinite(gross) & (gross > 0.1 * maximum))
    if indices.size == 0:
        raise FigureDataError("CBET transfer band is empty at the 10% threshold")
    return float(edges_cm[indices[0]]), float(edges_cm[indices[-1] + 1])


def _ray_path(trajectory: RayTrajectoryData, ray_index: int) -> RayPath | None:
    start = int(trajectory.offsets[ray_index])
    end = int(trajectory.offsets[ray_index + 1])
    if end - start < 2:
        return None
    pos_r = np.asarray(trajectory.pos_r_cm[start:end], dtype=float)
    pos_z = np.asarray(trajectory.pos_z_cm[start:end], dtype=float)
    power = np.asarray(trajectory.power_erg_s[start:end], dtype=float)
    valid = np.isfinite(pos_r) & np.isfinite(pos_z) & np.isfinite(power)
    if np.count_nonzero(valid) < 2:
        return None
    pos_r = pos_r[valid]
    pos_z = pos_z[valid]
    power = power[valid]

    ds = np.hypot(np.diff(pos_r), np.diff(pos_z))
    s_cm = np.concatenate(([0.0], np.cumsum(ds)))
    positive_power = power[power > 0.0]
    scale = float(np.max(positive_power)) if positive_power.size else 1.0
    floor = max(scale * 1.0e-300, np.finfo(float).tiny)
    log_power = np.log(np.maximum(power, floor))
    dlnp_ds = np.divide(
        np.diff(log_power),
        ds,
        out=np.full(ds.shape, np.nan),
        where=np.isfinite(ds) & (ds > 0.0),
    )
    return RayPath(
        index=ray_index,
        points_um=np.column_stack((pos_z, pos_r)) * CM_TO_UM,
        radius_cm=np.hypot(pos_r, pos_z),
        s_cm=s_cm,
        power_erg_s=power,
        dlnp_ds_cm=dlnp_ds,
    )


def _all_ray_paths(trajectory: RayTrajectoryData) -> list[RayPath]:
    paths: list[RayPath] = []
    for ray_index in range(trajectory.n_rays):
        path = _ray_path(trajectory, ray_index)
        if path is not None:
            paths.append(path)
    return paths


def _robust_symmetric_limit(values: np.ndarray) -> float:
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return 1.0
    limit = float(np.percentile(finite, 98.0))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = float(np.max(finite))
    return limit if limit > 0.0 else 1.0


def _segment_midpoint(path: RayPath, segment_index: int) -> np.ndarray:
    return 0.5 * (
        path.points_um[segment_index] + path.points_um[segment_index + 1]
    )


def _annotate_ray_events(
    ax: plt.Axes,
    paths: list[RayPath],
    band_cm: tuple[float, float],
) -> None:
    gain_candidates: list[tuple[float, RayPath, int]] = []
    loss_candidates: list[tuple[float, RayPath, int]] = []
    band_inner, band_outer = band_cm
    for path in paths:
        radius_mid = 0.5 * (path.radius_cm[:-1] + path.radius_cm[1:])
        radial_change = np.diff(path.radius_cm)
        in_band = (radius_mid >= band_inner) & (radius_mid <= band_outer)
        gain_indices = np.flatnonzero(
            np.isfinite(path.dlnp_ds_cm)
            & (path.dlnp_ds_cm > 0.0)
            & (radial_change > 0.0)
            & in_band
        )
        for index in gain_indices:
            gain_candidates.append(
                (float(path.dlnp_ds_cm[index]), path, int(index))
            )
        loss_indices = np.flatnonzero(
            np.isfinite(path.dlnp_ds_cm)
            & (path.dlnp_ds_cm < 0.0)
            & (radial_change < 0.0)
        )
        for index in loss_indices:
            loss_candidates.append(
                (float(-path.dlnp_ds_cm[index]), path, int(index))
            )

    if gain_candidates:
        _, path, index = max(gain_candidates, key=lambda item: item[0])
        xy = _segment_midpoint(path, index)
        ax.annotate(
            "出射光が増幅 = CBET",
            xy=xy,
            xytext=(18, 26),
            textcoords="offset points",
            color="#A00000",
            fontsize=10,
            arrowprops={
                "arrowstyle": "->",
                "color": "#A00000",
                "linewidth": 1.2,
            },
            zorder=12,
        )
    if loss_candidates:
        _, path, index = max(loss_candidates, key=lambda item: item[0])
        xy = _segment_midpoint(path, index)
        offset = (24, 28) if xy[0] < 0.0 else (-24, 28)
        ax.annotate(
            "IB+CBET 減衰",
            xy=xy,
            xytext=offset,
            textcoords="offset points",
            ha="left" if xy[0] < 0.0 else "right",
            color="#003B80",
            fontsize=10,
            arrowprops={
                "arrowstyle": "->",
                "color": "#003B80",
                "linewidth": 1.2,
            },
            zorder=12,
        )


def _band_crossings_s_cm(
    path: RayPath,
    band_cm: tuple[float, float],
) -> list[float]:
    crossings: list[float] = []
    for boundary in band_cm:
        for index in range(path.radius_cm.size - 1):
            r0 = float(path.radius_cm[index])
            r1 = float(path.radius_cm[index + 1])
            if not np.isfinite(r0) or not np.isfinite(r1) or r1 == r0:
                continue
            if (r0 - boundary) * (r1 - boundary) <= 0.0:
                fraction = (boundary - r0) / (r1 - r0)
                if 0.0 <= fraction <= 1.0:
                    crossings.append(
                        float(
                            path.s_cm[index]
                            + fraction
                            * (path.s_cm[index + 1] - path.s_cm[index])
                        )
                    )
    return sorted(set(crossings))


def _plot_power_inset(
    ax: plt.Axes,
    all_paths: list[RayPath],
    band_cm: tuple[float, float],
) -> None:
    scored: list[tuple[float, float, RayPath]] = []
    for path in all_paths:
        power_rise = np.diff(path.power_erg_s)
        positive_rise = (
            float(np.max(power_rise))
            if power_rise.size and np.any(power_rise > 0.0)
            else 0.0
        )
        finite_gradient = np.abs(
            path.dlnp_ds_cm[np.isfinite(path.dlnp_ds_cm)]
        )
        max_gradient = (
            float(np.max(finite_gradient)) if finite_gradient.size else 0.0
        )
        scored.append((positive_rise, max_gradient, path))

    has_gain = any(score[0] > 0.0 for score in scored)
    if has_gain:
        chosen = sorted(scored, key=lambda item: item[0], reverse=True)[:3]
    else:
        chosen = sorted(scored, key=lambda item: item[1], reverse=True)[:3]

    colors = ["#0072B2", "#D55E00", "#009E73"]
    for color, (_, _, path) in zip(colors, chosen):
        s_um = path.s_cm * CM_TO_UM
        power_tw = path.power_erg_s * ERG_PER_S_TO_TW
        ax.plot(
            s_um,
            power_tw,
            color=color,
            linewidth=1.6,
            label=f"ray {path.index}",
        )
        for crossing_s_cm in _band_crossings_s_cm(path, band_cm):
            crossing_s_um = crossing_s_cm * CM_TO_UM
            crossing_power_tw = float(
                np.interp(crossing_s_cm, path.s_cm, power_tw)
            )
            ax.scatter(
                [crossing_s_um],
                [crossing_power_tw],
                color=color,
                marker="x",
                s=28,
                linewidths=1.2,
                zorder=5,
            )

    if not has_gain:
        ax.text(
            0.5,
            0.06,
            "増幅区間なし(この時刻)",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            color="#8B0000",
            fontsize=9,
        )
    ax.set_title(
        "経路に沿う ray power\n— 上昇区間が交換の証拠",
        fontsize=10,
    )
    ax.set_xlabel(r"経路長 $s$ [$\mu$m]")
    ax.set_ylabel("P [TW]")
    ax.grid(alpha=0.22)
    if chosen:
        ax.legend(fontsize=8, loc="best")


def _plot_cbet_figure(
    snapshot: SnapshotRef,
    config: dict[str, object] | None,
    out_dir: Path,
    dpi: int,
    max_rays: int,
) -> bool:
    output_path = out_dir / f"cbet_rays_t{_time_tag(snapshot.time_s)}ns.png"
    hydro_centers_cm: np.ndarray | None = None
    hydro_ne_cm3: np.ndarray | None = None
    hydro_n_crit_cm3: float | None = None
    try:
        with h5py.File(snapshot.path, "r") as handle:
            if TRAJECTORY_GROUP not in handle:
                raise FigureDataError(
                    "missing '/laser/rays/trajectory'; enable "
                    "Laser.ray_output_trajectory (ray_output_trajectory)"
                )
            mesh: LaserMeshData = _read_mesh(handle)
            trajectory = _read_trajectory(handle)
            gross = np.asarray(
                _require_array(handle, HYDRO_CBET_GROSS),
                dtype=float,
            ).reshape(-1)
            edges_cm = np.asarray(
                _require_array(handle, MESH_X_R),
                dtype=float,
            ).reshape(-1)
            if config is not None:
                try:
                    hydro_centers_cm, hydro_ne_cm3 = (
                        _read_hydro_electron_density(handle, config)
                    )
                    hydro_n_crit_cm3 = _critical_density_cm3(config)
                except FigureDataError:
                    pass
    except (OSError, SnapshotFormatError, FigureDataError) as exc:
        _warn(f"skipping {output_path.name}: {exc}")
        return False

    try:
        band_cm = _cbet_band(edges_cm, gross)
    except FigureDataError as exc:
        _warn(f"skipping {output_path.name}: {exc}")
        return False
    if trajectory.n_rays <= 0:
        _warn(f"skipping {output_path.name}: trajectory contains no rays")
        return False

    all_paths = _all_ray_paths(trajectory)
    if not all_paths:
        _warn(f"skipping {output_path.name}: trajectory contains no valid ray paths")
        return False
    path_by_index = {path.index: path for path in all_paths}
    selected_paths = [
        path_by_index[int(index)]
        for index in _subsample_indices(trajectory.n_rays, max_rays)
        if int(index) in path_by_index
    ]

    derivative_values = np.concatenate(
        [
            path.dlnp_ds_cm[np.isfinite(path.dlnp_ds_cm)]
            for path in selected_paths
            if np.any(np.isfinite(path.dlnp_ds_cm))
        ]
    ) if any(np.any(np.isfinite(path.dlnp_ds_cm)) for path in selected_paths) else np.asarray([])
    absg = np.abs(
        derivative_values[
            np.isfinite(derivative_values) & (derivative_values != 0.0)
        ]
    )
    if absg.size:
        derivative_limit = float(np.percentile(absg, 99.5))
        # linthresh = p60 so weak CBET slopes stay visible against the ~1e3 IB cliff.
        linear_threshold = max(
            float(np.percentile(absg, 60.0)),
            1.0e-3 * derivative_limit,
        )
        derivative_norm = SymLogNorm(
            linthresh=linear_threshold,
            linscale=1.0,
            vmin=-derivative_limit,
            vmax=derivative_limit,
            base=10,
        )
    else:
        derivative_norm = None

    fig = plt.figure(figsize=(14.8, 7.6))
    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=(4.2, 1.35),
        height_ratios=(1.0, 0.07),
    )
    ax = fig.add_subplot(outer[0, 0])
    power_ax = fig.add_subplot(outer[0, 1])
    colorbar_grid = outer[1, 0].subgridspec(1, 2, wspace=0.32)
    density_cax = fig.add_subplot(colorbar_grid[0, 0])
    ray_cax = fig.add_subplot(colorbar_grid[0, 1])

    z_grid, r_grid = np.meshgrid(
        mesh.node_z_cm * CM_TO_UM,
        mesh.node_r_cm * CM_TO_UM,
    )
    density = np.ma.masked_less_equal(mesh.n_e_hat, 0.0)
    density_mesh = ax.pcolormesh(
        z_grid,
        r_grid,
        density,
        shading="gouraud",
        cmap="Greys",
        norm=LogNorm(vmin=1.0e-3, vmax=2.0),
        alpha=0.58,
        zorder=1,
    )
    fig.colorbar(
        density_mesh,
        cax=density_cax,
        orientation="horizontal",
        label=r"$n_e/n_c$",
    )

    density_min = float(np.nanmin(mesh.n_e_hat))
    density_max = float(np.nanmax(mesh.n_e_hat))
    if density_min <= 0.25 <= density_max:
        ax.contour(
            z_grid,
            r_grid,
            mesh.n_e_hat,
            levels=[0.25],
            colors="#404040",
            linestyles="--",
            linewidths=1.2,
            zorder=4,
        )
    else:
        hydro_radius = None
        if (
            hydro_centers_cm is not None
            and hydro_ne_cm3 is not None
            and hydro_n_crit_cm3 is not None
        ):
            indices = np.flatnonzero(
                np.isfinite(hydro_ne_cm3)
                & (hydro_ne_cm3 >= 0.25 * hydro_n_crit_cm3)
            )
            if indices.size:
                hydro_radius = float(hydro_centers_cm[indices[-1]])
        if hydro_radius is not None:
            ax.add_patch(
                plt.Circle(
                    (0.0, 0.0),
                    hydro_radius * CM_TO_UM,
                    fill=False,
                    edgecolor="#404040",
                    linestyle="--",
                    linewidth=1.2,
                    zorder=4,
                )
            )
            _warn(
                f"{output_path.name}: n_c/4 contour is outside the "
                "laser-mesh range -> hydro fallback used"
            )
        else:
            _warn(
                f"{output_path.name}: n_c/4 contour is outside the laser-mesh range"
            )
    if density_min <= 1.0 <= density_max:
        ax.contour(
            z_grid,
            r_grid,
            mesh.n_e_hat,
            levels=[1.0],
            colors="#202020",
            linestyles="-",
            linewidths=1.4,
            zorder=4,
        )
    else:
        hydro_radius = None
        if (
            hydro_centers_cm is not None
            and hydro_ne_cm3 is not None
            and hydro_n_crit_cm3 is not None
        ):
            indices = np.flatnonzero(
                np.isfinite(hydro_ne_cm3)
                & (hydro_ne_cm3 >= hydro_n_crit_cm3)
            )
            if indices.size:
                hydro_radius = float(hydro_centers_cm[indices[-1]])
        if hydro_radius is not None:
            ax.add_patch(
                plt.Circle(
                    (0.0, 0.0),
                    hydro_radius * CM_TO_UM,
                    fill=False,
                    edgecolor="#202020",
                    linestyle="-",
                    linewidth=1.4,
                    zorder=4,
                )
            )
            _warn(
                f"{output_path.name}: n_c contour is outside the "
                "laser-mesh range -> hydro fallback used"
            )
        else:
            _warn(f"{output_path.name}: n_c contour is outside the laser-mesh range")

    band_inner_um = band_cm[0] * CM_TO_UM
    band_outer_um = band_cm[1] * CM_TO_UM
    ax.add_patch(
        Wedge(
            (0.0, 0.0),
            band_outer_um,
            0.0,
            360.0,
            width=max(band_outer_um - band_inner_um, 1.0e-12),
            facecolor="#D62728",
            edgecolor="none",
            alpha=0.18,
            zorder=2,
        )
    )

    segment_sets: list[np.ndarray] = []
    segment_values: list[np.ndarray] = []
    for path in selected_paths:
        segments = np.stack(
            (path.points_um[:-1], path.points_um[1:]),
            axis=1,
        )
        valid = np.isfinite(path.dlnp_ds_cm)
        if np.any(valid):
            segment_sets.append(segments[valid])
            segment_values.append(path.dlnp_ds_cm[valid])

    ray_collection = None
    if segment_sets and derivative_norm is not None:
        ray_collection = LineCollection(
            np.concatenate(segment_sets, axis=0),
            cmap="coolwarm",
            norm=derivative_norm,
            linewidths=2.2,
            capstyle="round",
            alpha=0.95,
            zorder=6,
        )
        ray_collection.set_array(np.concatenate(segment_values))
        ray_collection.set_path_effects(
            [
                pe.Stroke(linewidth=3.4, foreground="white", alpha=0.65),
                pe.Normal(),
            ]
        )
        ax.add_collection(ray_collection)
        fig.colorbar(
            ray_collection,
            cax=ray_cax,
            orientation="horizontal",
            label=r"$d(\ln P)/ds$ [cm$^{-1}$]  青: 減衰 / 赤: 増幅",
        )
    else:
        ray_cax.axis("off")
        _warn(f"{output_path.name}: no valid ray segments selected by --max-rays")

    _annotate_ray_events(ax, selected_paths, band_cm)
    _plot_power_inset(power_ax, all_paths, band_cm)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#202020",
            linestyle="-",
            linewidth=1.4,
            label="臨界面",
        ),
        Line2D(
            [0],
            [0],
            color="#404040",
            linestyle="--",
            linewidth=1.2,
            label="1/4 臨界",
        ),
        Patch(
            facecolor="#D62728",
            edgecolor="none",
            alpha=0.18,
            label="CBET 転送帯",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
    ax.set_xlabel(r"$Z$ [$\mu$m]")
    ax.set_ylabel(r"$R$ [$\mu$m]")
    ax.set_title(
        f"CBET エネルギー交換  —  t = {_time_tag(snapshot.time_s)} ns"
    )
    ax.set_xlim(
        float(np.nanmin(mesh.node_z_cm)) * CM_TO_UM,
        float(np.nanmax(mesh.node_z_cm)) * CM_TO_UM,
    )
    ax.set_ylim(
        float(np.nanmin(mesh.node_r_cm)) * CM_TO_UM,
        float(np.nanmax(mesh.node_r_cm)) * CM_TO_UM,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-200.0, 200.0)
    ax.set_ylim(0.0, 200.0)
    ax.tick_params(direction="in")
    power_ax.tick_params(direction="in")
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.13,
        top=0.91,
        wspace=0.18,
        hspace=0.55,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {snapshot.path} -> {output_path}")
    return True


def _normalized_ports(config: dict[str, object]) -> dict[str, object]:
    laser = config.get("laser")
    port_configuration = (
        laser.get("port_configuration") if isinstance(laser, dict) else None
    )
    port_table = (
        port_configuration.get("ports")
        if isinstance(port_configuration, dict)
        else None
    )
    if not isinstance(port_table, list) or not port_table:
        raise FigureDataError(
            "missing frozen config field 'laser.port_configuration.ports'"
        )
    axes: list[np.ndarray] = []
    ids: list[object] = []
    for index, port in enumerate(port_table):
        if not isinstance(port, dict):
            raise FigureDataError("invalid frozen port entry")
        direction = np.asarray(port.get("direction", []), dtype=float)
        norm = float(np.linalg.norm(direction))
        if (
            direction.shape != (3,)
            or not np.all(np.isfinite(direction))
            or norm <= 0.0
        ):
            raise FigureDataError(
                "frozen port directions must be finite nonzero 3-vectors"
            )
        axes.append(direction / norm)
        ids.append(port.get("port_id", index))
    return {"axes": np.asarray(axes), "ids": ids}


def _frozen_channels(config: dict[str, object]) -> list[dict[str, object]]:
    laser = config.get("laser")
    hot_electron = laser.get("hot_electron") if isinstance(laser, dict) else None
    sources = hot_electron.get("sources") if isinstance(hot_electron, dict) else None
    if not isinstance(sources, list) or not sources:
        raise FigureDataError(
            "missing frozen config field 'laser.hot_electron.sources'"
        )
    if not all(isinstance(source, dict) for source in sources):
        raise FigureDataError("invalid frozen hot-electron source entry")
    return sources  # type: ignore[return-value]


def _channel_surfaces(
    channels: list[dict[str, object]],
    plasma: RadialPlasma,
) -> list[ChannelSurface]:
    result: list[ChannelSurface] = []
    for index, channel in enumerate(channels):
        mechanism = str(channel.get("mechanism", "")).lower()
        capture_fraction = channel.get("capture_nc_fraction")
        if (
            mechanism not in {"tpd", "srs", "cone"}
            or not isinstance(capture_fraction, (int, float))
            or not np.isfinite(capture_fraction)
        ):
            raise FigureDataError(
                f"invalid frozen hot-electron source at channel {index}"
            )
        radius = _outermost_crossing_radius(
            plasma.centers_cm,
            plasma.n_hat,
            float(capture_fraction),
        )
        if radius is None:
            raise FigureDataError(
                f"no n_e/n_c={float(capture_fraction):g} capture crossing "
                f"for channel {index} ({mechanism})"
            )
        if mechanism == "tpd":
            theta_center = channel.get("tpd_theta_deg")
            theta_delta = channel.get("tpd_delta_deg")
            if (
                not isinstance(theta_center, (int, float))
                or not isinstance(theta_delta, (int, float))
                or not np.isfinite(theta_center)
                or not np.isfinite(theta_delta)
            ):
                raise FigureDataError(
                    f"missing tpd_theta_deg/tpd_delta_deg for channel {index}"
                )
            angle_low = max(0.0, float(theta_center) - float(theta_delta))
            angle_high = min(180.0, float(theta_center) + float(theta_delta))
            angles = np.linspace(angle_low, angle_high, 5)
        else:
            theta_div = channel.get("theta_div_deg")
            if (
                not isinstance(theta_div, (int, float))
                or not np.isfinite(theta_div)
            ):
                raise FigureDataError(
                    f"missing theta_div_deg for channel {index} ({mechanism})"
                )
            angles = np.linspace(-float(theta_div), float(theta_div), 5)
        result.append(
            ChannelSurface(
                index=index,
                mechanism=mechanism,
                radius_cm=radius,
                angles_deg=angles,
            )
        )
    return result


def _select_meridian_ports(
    ports: dict[str, object],
) -> list[tuple[int, np.ndarray, float]]:
    axes = np.asarray(ports["axes"], dtype=float)
    meridian_normal = np.asarray([0.0, 1.0, 0.0])
    selected: list[tuple[int, np.ndarray, float]] = []
    band = np.radians(25.0)
    for port_index, axis in enumerate(axes):
        normal_component = float(np.dot(axis, meridian_normal))
        plane_angle = float(
            np.arcsin(np.clip(abs(normal_component), 0.0, 1.0))
        )
        if plane_angle > band:
            continue
        projected = axis - normal_component * meridian_normal
        projected_norm = float(np.linalg.norm(projected))
        if projected_norm <= 0.0:
            continue
        projected /= projected_norm
        # This is the +/-25 degree meridian selection used by
        # tools/plot_multibeam_lpi.py, including its raised-cosine weight.
        plane_weight = float(
            np.cos(0.5 * math.pi * plane_angle / band) ** 2
        )
        selected.append((port_index, projected, plane_weight))
    return selected


def _hot_e_r90(edges_cm: np.ndarray, hot_e_q: np.ndarray) -> float:
    if hot_e_q.size != edges_cm.size - 1:
        raise FigureDataError(
            "datasets '/hydro/hot_e_Q' and '/mesh/x_r' have incompatible sizes"
        )
    shell_volume = (
        4.0
        * math.pi
        / 3.0
        * (edges_cm[1:] ** 3 - edges_cm[:-1] ** 3)
    )
    shell_power = np.where(
        np.isfinite(hot_e_q) & (hot_e_q > 0.0),
        hot_e_q * shell_volume,
        0.0,
    )
    total = float(np.sum(shell_power))
    if not np.isfinite(total) or total <= 0.0:
        raise FigureDataError(
            "dataset '/hydro/hot_e_Q' has no positive volume-integrated deposition"
        )
    target = 0.9 * total
    cumulative_outside = 0.0
    for index in range(shell_power.size - 1, -1, -1):
        next_cumulative = cumulative_outside + float(shell_power[index])
        if next_cumulative >= target:
            needed = target - cumulative_outside
            q_value = float(hot_e_q[index])
            if q_value > 0.0:
                outer_cubed = float(edges_cm[index + 1] ** 3)
                radius_cubed = outer_cubed - needed / (
                    q_value * 4.0 * math.pi / 3.0
                )
                radius_cubed = float(
                    np.clip(
                        radius_cubed,
                        edges_cm[index] ** 3,
                        edges_cm[index + 1] ** 3,
                    )
                )
                return float(np.cbrt(radius_cubed))
            return float(edges_cm[index])
        cumulative_outside = next_cumulative
    return float(edges_cm[0])


def _rotate_2d(vector: np.ndarray, angle_deg: float) -> np.ndarray:
    angle = np.radians(angle_deg)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return np.asarray(
        [
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
        ]
    )


def _sphere_crossing_distance(
    point_cm: np.ndarray,
    direction: np.ndarray,
    radius_cm: float,
) -> float | None:
    projection = float(np.dot(point_cm, direction))
    discriminant = projection * projection + radius_cm**2 - float(
        np.dot(point_cm, point_cm)
    )
    if discriminant < 0.0:
        return None
    roots = [
        -projection - math.sqrt(max(discriminant, 0.0)),
        -projection + math.sqrt(max(discriminant, 0.0)),
    ]
    positive = [root for root in roots if root > 1.0e-14]
    return min(positive) if positive else None


def _chord_collection(
    start_cm: np.ndarray,
    direction: np.ndarray,
    source_radius_cm: float,
    r90_cm: float,
    color: str,
) -> tuple[LineCollection, np.ndarray]:
    s_ca = -float(np.dot(start_cm, direction))
    if s_ca <= 0.0:
        raise FigureDataError("hot-electron chord does not point inward")
    impact_parameter = math.sqrt(
        max(source_radius_cm**2 - s_ca**2, 0.0)
    )
    overshoot = (
        0.35 * (source_radius_cm - r90_cm)
        if r90_cm < source_radius_cm
        else 0.1 * source_radius_cm
    )
    if impact_parameter < r90_cm:
        s_stop = (
            s_ca
            - math.sqrt(r90_cm**2 - impact_parameter**2)
            + overshoot
        )
    else:
        s_stop = s_ca + overshoot
    s_stop = min(s_stop, s_ca + overshoot)
    s_stop = max(s_stop, 0.05 * source_radius_cm)

    distance = np.linspace(0.0, s_stop, 65)
    points_cm = start_cm[np.newaxis, :] + distance[:, np.newaxis] * direction
    points_um = points_cm * CM_TO_UM
    segments = np.stack((points_um[:-1], points_um[1:]), axis=1)
    alpha = np.linspace(1.0, 0.15, segments.shape[0])
    rgba = np.tile(matplotlib.colors.to_rgba(color), (segments.shape[0], 1))
    rgba[:, 3] = alpha
    collection = LineCollection(
        segments,
        colors=rgba,
        linewidths=1.0,
        zorder=5,
    )
    return collection, points_um


def _hot_e_rgba(
    edges_cm: np.ndarray,
    hot_e_q: np.ndarray,
    limit_cm: float,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    coordinates = np.linspace(-limit_cm, limit_cm, 600)
    z_grid, r_grid = np.meshgrid(coordinates, coordinates)
    radius = np.hypot(z_grid, r_grid)
    shell_index = np.searchsorted(edges_cm, radius, side="right") - 1
    valid = (
        (shell_index >= 0)
        & (shell_index < hot_e_q.size)
        & (radius <= edges_cm[-1])
    )
    values = np.zeros_like(radius)
    values[valid] = np.maximum(hot_e_q[shell_index[valid]], 0.0)
    positive = hot_e_q[np.isfinite(hot_e_q) & (hot_e_q > 0.0)]
    log_positive = np.log10(positive)
    log_low = float(np.percentile(log_positive, 2.0))
    log_high = float(np.percentile(log_positive, 98.0))
    if not log_high > log_low:
        log_low = log_high - 1.0
    normalized = np.zeros_like(values)
    positive_grid = values > 0.0
    normalized[positive_grid] = np.clip(
        (np.log10(values[positive_grid]) - log_low) / (log_high - log_low),
        0.0,
        1.0,
    )
    rgba = plt.get_cmap("Oranges")(0.25 + 0.75 * normalized)
    rgba[..., 3] = np.where(positive_grid, 0.08 + 0.72 * normalized, 0.0)
    extent_um = (
        -limit_cm * CM_TO_UM,
        limit_cm * CM_TO_UM,
        -limit_cm * CM_TO_UM,
        limit_cm * CM_TO_UM,
    )
    return rgba, extent_um


def _plot_hot_e_figure(
    snapshot: SnapshotRef,
    config: dict[str, object] | None,
    out_dir: Path,
    dpi: int,
) -> bool:
    output_path = out_dir / f"hote_paths_t{_time_tag(snapshot.time_s)}ns.png"
    if config is None:
        _warn(f"skipping {output_path.name}: frozen config is unavailable")
        return False

    try:
        ports = _normalized_ports(config)
        channels = _frozen_channels(config)
        with h5py.File(snapshot.path, "r") as handle:
            plasma = _read_radial_plasma(handle, config)
            hot_e_q = np.asarray(
                _require_array(handle, HYDRO_HOT_E_Q),
                dtype=float,
            ).reshape(-1)
            capture_power = np.asarray(
                _require_array(handle, PORT_CAPTURE_PCROSS),
                dtype=float,
            )
        surfaces = _channel_surfaces(channels, plasma)
        if hot_e_q.size != plasma.centers_cm.size:
            raise FigureDataError(
                "dataset '/hydro/hot_e_Q' has an incompatible profile size"
            )
        n_ports = np.asarray(ports["axes"]).shape[0]
        if capture_power.ndim != 2 or capture_power.shape != (
            n_ports,
            len(channels),
        ):
            raise FigureDataError(
                "dataset '/laser/port_section/port_capture_Pcross' must have "
                f"shape ({n_ports}, {len(channels)}), got {capture_power.shape}"
            )
        r90_cm = _hot_e_r90(plasma.edges_cm, hot_e_q)
        selected_ports = _select_meridian_ports(ports)
        if not selected_ports:
            raise FigureDataError(
                "no port axes lie within +/-25 deg of the phi=0 meridian"
            )
        selected_capture = np.asarray(
            [
                capture_power[port_index, surface.index]
                for surface in surfaces
                for port_index, _, _ in selected_ports
            ],
            dtype=float,
        )
        positive_capture = selected_capture[
            np.isfinite(selected_capture) & (selected_capture > 0.0)
        ]
        if positive_capture.size == 0:
            raise FigureDataError(
                "selected '/laser/port_section/port_capture_Pcross' values "
                "contain no positive captured power"
            )
        capture_max = float(np.max(positive_capture))
    except (OSError, FigureDataError) as exc:
        _warn(f"skipping {output_path.name}: {exc}")
        return False

    critical_radii = {
        0.25: _outermost_crossing_radius(
            plasma.centers_cm,
            plasma.n_hat,
            0.25,
        ),
        1.0: _outermost_crossing_radius(
            plasma.centers_cm,
            plasma.n_hat,
            1.0,
        ),
    }
    for level, radius in critical_radii.items():
        if radius is None:
            _warn(
                f"{output_path.name}: n_e/n_c={level:g} contour is not crossed"
            )

    scale_cm = max(
        [surface.radius_cm for surface in surfaces]
        + [r90_cm]
        + [
            radius
            for radius in critical_radii.values()
            if radius is not None
        ]
    )
    limit_cm = 1.18 * scale_cm
    rgba, extent_um = _hot_e_rgba(
        plasma.edges_cm,
        hot_e_q,
        limit_cm,
    )

    fig, ax = plt.subplots(figsize=(9.2, 8.3))
    fig.subplots_adjust(bottom=0.10)
    ax.imshow(
        rgba,
        origin="lower",
        extent=extent_um,
        interpolation="bilinear",
        zorder=1,
    )

    angle_grid = np.linspace(0.0, 2.0 * math.pi, 721)
    channel_colors = ["#0072B2", "#CC79A7", "#009E73", "#E69F00"]
    legend_handles: list[object] = [
        Patch(
            facecolor="#E66101",
            edgecolor="none",
            alpha=0.55,
            label="沈着帯",
        )
    ]
    for level, radius in critical_radii.items():
        if radius is None:
            continue
        radius_um = radius * CM_TO_UM
        linestyle = "--" if level == 0.25 else "-"
        ax.plot(
            radius_um * np.cos(angle_grid),
            radius_um * np.sin(angle_grid),
            color="#666666",
            linestyle=linestyle,
            linewidth=0.8,
            zorder=2,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#666666",
                linestyle=linestyle,
                linewidth=0.8,
                label=r"$n_e/n_c=$" + f"{level:g}",
            )
        )

    first_chord_points: np.ndarray | None = None
    first_generation_point: np.ndarray | None = None
    for surface_number, surface in enumerate(surfaces):
        color = channel_colors[surface_number % len(channel_colors)]
        radius_um = surface.radius_cm * CM_TO_UM
        mechanism_label = surface.mechanism.upper()
        ax.plot(
            radius_um * np.cos(angle_grid),
            radius_um * np.sin(angle_grid),
            color=color,
            linestyle="--",
            linewidth=1.2,
            zorder=3,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle="--",
                linewidth=1.2,
                label=f"生成面 {mechanism_label}",
            )
        )

        for port_index, projected, plane_weight in selected_ports:
            # Display coordinates follow plot_laser_rays.py: (Z, signed R).
            outward = np.asarray([projected[2], projected[0]])
            outward /= np.linalg.norm(outward)
            generation_cm = surface.radius_cm * outward
            capture_value = float(capture_power[port_index, surface.index])
            marker_size = (
                180.0 * max(capture_value, 0.0) / capture_max
                if np.isfinite(capture_value)
                else 0.0
            )
            ax.scatter(
                [generation_cm[0] * CM_TO_UM],
                [generation_cm[1] * CM_TO_UM],
                s=marker_size,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                alpha=0.35 + 0.65 * plane_weight,
                zorder=8,
            )
            if first_generation_point is None:
                first_generation_point = generation_cm * CM_TO_UM

            inward = -outward
            for angle_deg in surface.angles_deg:
                # Angles are measured from the inward radial direction and
                # rotated in the displayed meridian plane.
                direction = _rotate_2d(inward, float(angle_deg))
                collection, points_um = _chord_collection(
                    generation_cm,
                    direction,
                    surface.radius_cm,
                    r90_cm,
                    color,
                )
                ax.add_collection(collection)
                if first_chord_points is None:
                    first_chord_points = points_um

    marker_legend = ax.scatter(
        [],
        [],
        s=95,
        facecolor="#666666",
        edgecolor="white",
        linewidth=0.6,
        label="点サイズ ∝ 捕獲パワー",
    )
    legend_handles.append(marker_legend)

    if first_chord_points is not None:
        label_index = min(
            first_chord_points.shape[0] - 1,
            max(1, first_chord_points.shape[0] // 3),
        )
        ax.annotate(
            "CSDA 飛翔チョード",
            xy=first_chord_points[label_index],
            xytext=(18, 18),
            textcoords="offset points",
            fontsize=9,
            arrowprops={
                "arrowstyle": "->",
                "color": "#333333",
                "linewidth": 1.0,
            },
            zorder=12,
        )

    positive_q_indices = np.flatnonzero(
        np.isfinite(hot_e_q) & (hot_e_q > 0.0)
    )
    peak_q_index = int(positive_q_indices[np.argmax(hot_e_q[positive_q_indices])])
    deposition_point_um = np.asarray(
        [0.0, plasma.centers_cm[peak_q_index] * CM_TO_UM]
    )
    if first_generation_point is not None and first_chord_points is not None:
        flight_point = first_chord_points[first_chord_points.shape[0] // 2]
        ax.annotate(
            "生成 (捕獲面)",
            xy=first_generation_point,
            xytext=(0.05, 0.93),
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "#333333"},
            fontsize=10,
            zorder=12,
        )
        ax.annotate(
            "飛翔",
            xy=flight_point,
            xytext=(0.42, 0.93),
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "#333333"},
            fontsize=10,
            zorder=12,
        )
        ax.annotate(
            "沈着 (高密度殻)",
            xy=deposition_point_um,
            xytext=(0.71, 0.93),
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "#333333"},
            fontsize=10,
            zorder=12,
        )
        ax.text(
            0.33,
            0.935,
            "→",
            transform=ax.transAxes,
            fontsize=13,
            ha="center",
            va="center",
        )
        ax.text(
            0.64,
            0.935,
            "→",
            transform=ax.transAxes,
            fontsize=13,
            ha="center",
            va="center",
        )

    ax.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=8,
        ncols=2,
    )
    ax.set_xlim(-200.0, 200.0)
    ax.set_ylim(-200.0, 200.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$Z$ [$\mu$m]")
    ax.set_ylabel(r"signed $R$ [$\mu$m]")
    ax.set_title(
        f"hot-e 生成→飛翔→沈着  —  t = {_time_tag(snapshot.time_s)} ns"
    )
    ax.tick_params(direction="in")
    fig.text(
        0.5,
        0.012,
        "※ チョード方向は生成点の内向き半径方向を基準に面内回転。"
        "1D 輸送は方位角対称。",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"[ok] {snapshot.path} -> {output_path}")
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot CBET energy-exchange rays and hot-electron "
            "generation/flight/deposition paths."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="TENRYU run directory containing results/ and config/.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: "
            "~/work/projects/TENRYU/outputs/plots/<run-basename>_rays/"
        ),
    )
    parser.add_argument(
        "--times",
        type=str,
        default=None,
        help=(
            "Comma-separated snapshot times in ns. Default: snapshots nearest "
            "0.5 and 0.95 of the last snapshot time."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output PNG DPI (default: 150).",
    )
    parser.add_argument(
        "--max-rays",
        type=int,
        default=24,
        help="Maximum evenly subsampled ray trajectories in Figure A (default: 24).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_dependencies()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error("--dpi must be > 0")
    if args.max_rays < 0:
        parser.error("--max-rays must be >= 0")
    if not args.run_dir.is_dir():
        parser.error(f"RUN_DIR is not a directory: {args.run_dir}")

    requested_times_ns: list[float] | None = None
    if args.times is not None:
        try:
            requested_times_ns = _parse_times_ns(args.times)
        except ValueError as exc:
            parser.error(str(exc))

    try:
        snapshots = _discover_snapshots(args.run_dir)
    except (FileNotFoundError, OSError) as exc:
        parser.error(str(exc))
    selected = _select_snapshots(snapshots, requested_times_ns)
    config, _ = _load_frozen_config(args.run_dir)
    out_dir = (
        args.out
        if args.out is not None
        else Path("~/work/projects/TENRYU/outputs/plots").expanduser()
        / f"{args.run_dir.resolve().name}_rays"
    )

    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "axes.unicode_minus": False,
            "figure.dpi": args.dpi,
            "savefig.dpi": args.dpi,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
        }
    )

    for snapshot in selected:
        _plot_cbet_figure(
            snapshot,
            config,
            out_dir,
            args.dpi,
            args.max_rays,
        )
        _plot_hot_e_figure(
            snapshot,
            config,
            out_dir,
            args.dpi,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
