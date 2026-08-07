#!/usr/bin/env python3
"""Plot TENRYU laser ray trajectories over electron density maps from snapshot HDF5."""

from __future__ import annotations

import argparse
import json
import os
import re
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
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ModuleNotFoundError:  # pragma: no cover - dependency guard
    plt = None  # type: ignore[assignment]
    LineCollection = None  # type: ignore[assignment]
    Normalize = None  # type: ignore[assignment]
    make_axes_locatable = None  # type: ignore[assignment]


SNAPSHOT_GLOB = "*_[0-9][0-9][0-9][0-9].h5"
SNAPSHOT_INDEX_RE = re.compile(r"_(\d{4})\.h5$")


@dataclass
class LaserMeshData:
    node_r_cm: np.ndarray
    node_z_cm: np.ndarray
    n_e_hat: np.ndarray
    n_crit_cm3: float


@dataclass
class RayTrajectoryData:
    n_rays: int
    offsets: np.ndarray
    step_count: np.ndarray
    beam_id: np.ndarray
    pos_r_cm: np.ndarray
    pos_z_cm: np.ndarray
    power_erg_s: np.ndarray


@dataclass
class SnapshotData:
    name: str
    geometry: str
    time_s: float
    mesh: LaserMeshData
    trajectory: RayTrajectoryData


class SnapshotFormatError(RuntimeError):
    """Raised when snapshot layout is incompatible with expected laser output schema."""


def _ensure_deps() -> None:
    if h5py is None:
        raise RuntimeError("h5py is required. Install dependency: pip install h5py")
    if plt is None:
        raise RuntimeError("matplotlib is required. Install dependency: pip install matplotlib")


def _decode_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise SnapshotFormatError(f"expected scalar text value, got shape {value.shape}")
        return _decode_text(value.item())
    if isinstance(value, np.generic):
        return _decode_text(value.item())
    raise SnapshotFormatError(f"expected text value, got {type(value).__name__}")


def _read_attr_float(h5: h5py.File, key: str) -> float:
    if key not in h5.attrs:
        raise SnapshotFormatError(f"missing required root attribute '{key}'")
    raw = np.asarray(h5.attrs[key])
    if raw.shape != ():
        raise SnapshotFormatError(f"root attribute '{key}' must be scalar, got shape {raw.shape}")
    return float(raw.item())


def _read_attr_text(h5: h5py.File, key: str) -> str:
    if key not in h5.attrs:
        raise SnapshotFormatError(f"missing required root attribute '{key}'")
    return _decode_text(h5.attrs[key])


def _require_dataset(h5: h5py.File, path: str) -> h5py.Dataset:
    obj = h5.get(path)
    if obj is None:
        raise SnapshotFormatError(f"missing required dataset '{path}'")
    if not isinstance(obj, h5py.Dataset):
        raise SnapshotFormatError(f"expected dataset at '{path}', got {type(obj).__name__}")
    return obj


def _read_scalar_int(h5: h5py.File, path: str) -> int:
    ds = _require_dataset(h5, path)
    raw = np.asarray(ds[...])
    if raw.shape not in ((), (1,)):
        raise SnapshotFormatError(f"dataset '{path}' must be scalar, got shape {raw.shape}")
    return int(raw.reshape(-1)[0])


def _read_scalar_float(h5: h5py.File, path: str) -> float:
    ds = _require_dataset(h5, path)
    raw = np.asarray(ds[...], dtype=np.float64)
    if raw.shape not in ((), (1,)):
        raise SnapshotFormatError(f"dataset '{path}' must be scalar, got shape {raw.shape}")
    return float(raw.reshape(-1)[0])


def _read_vector_float(h5: h5py.File, path: str) -> np.ndarray:
    ds = _require_dataset(h5, path)
    arr = np.asarray(ds[...], dtype=np.float64)
    if arr.ndim != 1:
        raise SnapshotFormatError(f"dataset '{path}' must be rank-1, got rank {arr.ndim}")
    return arr


def _read_vector_int(h5: h5py.File, path: str, dtype: np.dtype) -> np.ndarray:
    ds = _require_dataset(h5, path)
    arr = np.asarray(ds[...], dtype=dtype)
    if arr.ndim != 1:
        raise SnapshotFormatError(f"dataset '{path}' must be rank-1, got rank {arr.ndim}")
    return arr


def _read_mesh(h5: h5py.File) -> LaserMeshData:
    nr = _read_scalar_int(h5, "laser/mesh/n_nodes_r")
    nz = _read_scalar_int(h5, "laser/mesh/n_nodes_z")
    if nr <= 0 or nz <= 0:
        raise SnapshotFormatError(
            f"laser mesh dimensions must be positive, got n_nodes_r={nr}, n_nodes_z={nz}"
        )

    node_r = _read_vector_float(h5, "laser/mesh/node_R")
    node_z = _read_vector_float(h5, "laser/mesh/node_Z")
    if node_r.size != nr:
        raise SnapshotFormatError(
            f"dataset 'laser/mesh/node_R' size mismatch: got {node_r.size}, expected {nr}"
        )
    if node_z.size != nz:
        raise SnapshotFormatError(
            f"dataset 'laser/mesh/node_Z' size mismatch: got {node_z.size}, expected {nz}"
        )

    ds_n_e_hat = _require_dataset(h5, "laser/mesh/n_e_hat")
    n_e_hat_raw = np.asarray(ds_n_e_hat[...], dtype=np.float64)
    expected_n = nr * nz
    if n_e_hat_raw.ndim == 1:
        if n_e_hat_raw.size != expected_n:
            raise SnapshotFormatError(
                "dataset 'laser/mesh/n_e_hat' size mismatch: "
                f"got {n_e_hat_raw.size}, expected {expected_n}"
            )
        n_e_hat = n_e_hat_raw.reshape((nr, nz), order="C")
    elif n_e_hat_raw.ndim == 2 and tuple(n_e_hat_raw.shape) == (nr, nz):
        n_e_hat = n_e_hat_raw
    else:
        raise SnapshotFormatError(
            "dataset 'laser/mesh/n_e_hat' must be shape "
            f"({expected_n},) or ({nr}, {nz}), got {tuple(n_e_hat_raw.shape)}"
        )

    n_crit = _read_scalar_float(h5, "laser/mesh/n_crit")
    return LaserMeshData(
        node_r_cm=node_r,
        node_z_cm=node_z,
        n_e_hat=n_e_hat,
        n_crit_cm3=n_crit,
    )


def _read_trajectory(h5: h5py.File) -> RayTrajectoryData:
    n_rays = _read_scalar_int(h5, "laser/rays/trajectory/n_rays")
    if n_rays < 0:
        raise SnapshotFormatError(f"dataset 'laser/rays/trajectory/n_rays' is negative: {n_rays}")

    offsets = _read_vector_int(h5, "laser/rays/trajectory/offsets", np.int64)
    step_count = _read_vector_int(h5, "laser/rays/trajectory/step_count", np.int64)
    beam_id = _read_vector_int(h5, "laser/rays/trajectory/beam_id", np.int64)

    if offsets.size != n_rays + 1:
        raise SnapshotFormatError(
            "dataset 'laser/rays/trajectory/offsets' size mismatch: "
            f"got {offsets.size}, expected {n_rays + 1}"
        )
    if step_count.size != n_rays:
        raise SnapshotFormatError(
            "dataset 'laser/rays/trajectory/step_count' size mismatch: "
            f"got {step_count.size}, expected {n_rays}"
        )
    if beam_id.size != n_rays:
        raise SnapshotFormatError(
            "dataset 'laser/rays/trajectory/beam_id' size mismatch: "
            f"got {beam_id.size}, expected {n_rays}"
        )
    if offsets.size > 0 and offsets[0] != 0:
        raise SnapshotFormatError(
            "dataset 'laser/rays/trajectory/offsets' must start at 0, "
            f"got {offsets[0]}"
        )
    if np.any(offsets[1:] < offsets[:-1]):
        raise SnapshotFormatError("dataset 'laser/rays/trajectory/offsets' must be non-decreasing")
    if np.any(step_count < 0):
        raise SnapshotFormatError("dataset 'laser/rays/trajectory/step_count' contains negative values")

    total_steps = int(offsets[-1]) if offsets.size > 0 else 0
    if int(step_count.sum()) != total_steps:
        raise SnapshotFormatError(
            "trajectory CSR mismatch: sum(step_count) "
            f"({int(step_count.sum())}) != offsets[-1] ({total_steps})"
        )

    if "laser/rays/trajectory/pos_R" in h5 and "laser/rays/trajectory/pos_Z" in h5:
        pos_r = _read_vector_float(h5, "laser/rays/trajectory/pos_R")
        pos_z = _read_vector_float(h5, "laser/rays/trajectory/pos_Z")
    elif "laser/rays/trajectory/pos_x" in h5 and "laser/rays/trajectory/pos_y" in h5:
        pos_r = _read_vector_float(h5, "laser/rays/trajectory/pos_x")
        pos_z = _read_vector_float(h5, "laser/rays/trajectory/pos_y")
    else:
        raise SnapshotFormatError(
            "missing required trajectory position datasets; expected "
            "'laser/rays/trajectory/pos_R'+'pos_Z' (or 'pos_x'+'pos_y')"
        )
    power = _read_vector_float(h5, "laser/rays/trajectory/power")

    for path, arr in (
        ("laser/rays/trajectory/pos_R|pos_x", pos_r),
        ("laser/rays/trajectory/pos_Z|pos_y", pos_z),
        ("laser/rays/trajectory/power", power),
    ):
        if arr.size != total_steps:
            raise SnapshotFormatError(
                f"dataset '{path}' size mismatch: got {arr.size}, expected {total_steps}"
            )

    return RayTrajectoryData(
        n_rays=n_rays,
        offsets=offsets,
        step_count=step_count,
        beam_id=beam_id,
        pos_r_cm=pos_r,
        pos_z_cm=pos_z,
        power_erg_s=power,
    )


def _find_domain_range_cm(
    h5_path: Path,
    reference_path: Path | None,
) -> tuple[float, float] | None:
    """Determine (r_min, r_max) in cm for auto-zoom, using HDF5 data.

    Fallback chain:
      1. --reference file  ->  mesh/x_r from that file
      2. metadata/frozen_config JSON  ->  mesh.r_min, mesh.r_max
      3. mesh/x_r from h5_path
      4. laser/mesh/node_R from h5_path
    """
    if reference_path is not None:
        try:
            with h5py.File(reference_path, "r") as h5ref:
                xr = np.asarray(h5ref["mesh/x_r"][...], dtype=np.float64)
                return float(np.nanmin(xr)), float(np.nanmax(xr))
        except (OSError, KeyError) as exc:
            print(f"[warn] --reference {reference_path}: {exc}; falling back to auto", file=sys.stderr)

    try:
        with h5py.File(h5_path, "r") as h5:
            # Try 1: frozen_config JSON
            try:
                raw = h5["metadata/frozen_config"][()].decode("utf-8")
                cfg = json.loads(raw)
                r_min = float(cfg["mesh"]["r_min"])
                r_max = float(cfg["mesh"]["r_max"])
                if np.isfinite(r_min) and np.isfinite(r_max) and r_max > r_min:
                    return r_min, r_max
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass

            # Try 2: hydro mesh nodes
            try:
                xr = np.asarray(h5["mesh/x_r"][...], dtype=np.float64)
                return float(np.nanmin(xr)), float(np.nanmax(xr))
            except KeyError:
                pass

            # Try 3: laser mesh nodes
            try:
                nr = np.asarray(h5["laser/mesh/node_R"][...], dtype=np.float64)
                return float(np.nanmin(nr)), float(np.nanmax(nr))
            except KeyError:
                pass
    except OSError as exc:
        print(f"[warn] could not read {h5_path} for auto-zoom: {exc}", file=sys.stderr)

    return None


def _read_snapshot(path: Path) -> SnapshotData:
    if not path.exists():
        raise SnapshotFormatError(f"file not found: {path}")
    if not path.is_file():
        raise SnapshotFormatError(f"path is not a file: {path}")

    with h5py.File(path, "r") as h5:
        geometry = _read_attr_text(h5, "geometry")
        if "time" in h5.attrs:
            time_s = _read_attr_float(h5, "time")
        elif "t" in h5.attrs:
            time_s = _read_attr_float(h5, "t")
        else:
            raise SnapshotFormatError("missing required root attribute 'time' (or legacy 't')")

        mesh = _read_mesh(h5)
        trajectory = _read_trajectory(h5)

    return SnapshotData(
        name=path.stem,
        geometry=geometry,
        time_s=time_s,
        mesh=mesh,
        trajectory=trajectory,
    )


def _collect_snapshots_from_dir(dir_path: Path) -> list[Path]:
    results_dir = dir_path / "results"
    scan_dir = results_dir if results_dir.is_dir() else dir_path
    paths = []
    for path in scan_dir.glob(SNAPSHOT_GLOB):
        if not path.is_file():
            continue
        name = path.name
        if "_ckpt_" in name or "_history" in name:
            continue
        if SNAPSHOT_INDEX_RE.search(name) is None:
            continue
        paths.append(path)
    paths.sort(key=lambda path: (int(SNAPSHOT_INDEX_RE.search(path.name).group(1)), path.name))
    if not paths:
        raise FileNotFoundError(f"no snapshot files matched in {scan_dir}")
    return paths


def _subsample_indices(n_rays: int, max_rays: int) -> np.ndarray:
    if n_rays <= 0 or max_rays <= 0:
        return np.empty(0, dtype=np.int64)
    if n_rays <= max_rays:
        return np.arange(n_rays, dtype=np.int64)
    idx = np.linspace(0, n_rays - 1, max_rays, dtype=np.int64)
    return np.unique(idx)


def _default_output_path(input_path: Path, base_dir: Path | None = None) -> Path:
    out_dir = base_dir / "figs" / "rays" if base_dir is not None else input_path.parent / "laser_plot"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{input_path.stem}_laser_rays.pdf"


def _resolve_output_paths(
    inputs: list[Path],
    output: Path | None,
    base_dir: Path | None = None,
) -> list[Path]:
    if output is None:
        return [_default_output_path(path, base_dir) for path in inputs]
    if len(inputs) == 1:
        return [output]

    if output.exists():
        if not output.is_dir():
            raise ValueError(
                "--output must be a directory when multiple input files are provided"
            )
        out_dir = output
    else:
        if output.suffix:
            raise ValueError(
                "--output must be a directory path when multiple input files are provided"
            )
        out_dir = output
        out_dir.mkdir(parents=True, exist_ok=True)

    return [out_dir / f"{path.stem}_laser_rays.pdf" for path in inputs]


def _parse_figsize(value: str) -> tuple[float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError("--figsize must be formatted as width,height")
    try:
        width = float(parts[0])
        height = float(parts[1])
    except ValueError as exc:
        raise ValueError("--figsize values must be valid floats") from exc
    if width <= 0.0 or height <= 0.0:
        raise ValueError("--figsize width and height must be > 0")
    return width, height


def _parse_zoom(value: str) -> tuple[str, tuple[float, float, float, float] | None]:
    token = value.strip()
    lowered = token.lower()
    if lowered in ("auto", "full"):
        return lowered, None

    parts = [part.strip() for part in token.split(",")]
    if len(parts) != 4:
        raise ValueError(
            "--zoom must be 'auto', 'full', or 'Rmin,Rmax,Zmin,Zmax' in displayed axis units"
        )
    try:
        r_min, r_max, z_min, z_max = (float(part) for part in parts)
    except ValueError as exc:
        raise ValueError("--zoom manual bounds must be valid floats") from exc
    if r_max <= r_min:
        raise ValueError("--zoom manual bounds require Rmax > Rmin")
    if z_max <= z_min:
        raise ValueError("--zoom manual bounds require Zmax > Zmin")
    return "manual", (r_min, r_max, z_min, z_max)


def _resolve_plot_limits(
    snapshot: SnapshotData,
    args: argparse.Namespace,
    unit_scale: float,
    mesh_range_cm: tuple[float, float] | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return (x_limits, y_limits) = (Z_limits, R_limits)."""
    mesh_r = snapshot.mesh.node_r_cm * unit_scale
    mesh_z = snapshot.mesh.node_z_cm * unit_scale
    mesh_r_min = float(np.nanmin(mesh_r))
    mesh_r_max = float(np.nanmax(mesh_r))
    mesh_z_min = float(np.nanmin(mesh_z))
    mesh_z_max = float(np.nanmax(mesh_z))

    if args.zoom_mode == "full":
        return (mesh_z_min, mesh_z_max), (mesh_r_min, mesh_r_max)
    if args.zoom_mode == "manual":
        r_min, r_max, z_min, z_max = args.zoom_bounds
        return (z_min, z_max), (r_min, r_max)

    # Default "auto": fix to initial mesh range from namelist
    if mesh_range_cm is not None:
        r_min_disp = mesh_range_cm[0] * unit_scale
        r_max_disp = mesh_range_cm[1] * unit_scale
        return (-r_max_disp, r_max_disp), (r_min_disp, r_max_disp)

    # Fallback when namelist not found: adaptive from mesh + ray data
    traj = snapshot.trajectory
    if traj.n_rays > 0 and traj.pos_r_cm.size > 0:
        ray_z = traj.pos_z_cm * unit_scale
        ray_r = traj.pos_r_cm * unit_scale
        ray_z_min = float(np.nanmin(ray_z))
        ray_z_max = float(np.nanmax(ray_z))
        ray_r_max = float(np.nanmax(ray_r))
    else:
        ray_z_min = ray_z_max = ray_r_max = 0.0

    z_lo = min(mesh_z_min, ray_z_min)
    z_hi = max(mesh_z_max, ray_z_max)
    r_hi = max(mesh_r_max, ray_r_max)
    pad_z = (z_hi - z_lo) * 0.03
    pad_r = r_hi * 0.03
    return (z_lo - pad_z, z_hi + pad_z), (0.0, r_hi + pad_r)


def _plot_snapshot(
    snapshot: SnapshotData,
    args: argparse.Namespace,
    output_path: Path,
    mesh_range_cm: tuple[float, float] | None,
) -> None:
    unit_scale = 1.0e4 if args.um else 1.0
    unit_label = r"$\mu$m" if args.um else "cm"

    mesh = snapshot.mesh
    density_vmin = args.vmin if args.vmin is not None else (-3.0 if args.log_density else None)
    density_vmax = args.vmax if args.vmax is not None else (0.0 if args.log_density else None)

    plt.rcParams.update(
        {
            "figure.dpi": args.dpi,
            "savefig.dpi": args.dpi,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "lines.linewidth": 1.2,
        }
    )
    x_limits, y_limits = _resolve_plot_limits(snapshot, args, unit_scale, mesh_range_cm)

    node_z_scaled = mesh.node_z_cm * unit_scale
    node_r_scaled = mesh.node_r_cm * unit_scale

    # Clip mesh to the visible region plus a 1-cell halo to avoid large off-screen meshes.
    j0 = max(0, np.searchsorted(node_z_scaled, x_limits[0], side="left") - 1)
    j1 = min(len(node_z_scaled), np.searchsorted(node_z_scaled, x_limits[1], side="right") + 1)
    i0 = max(0, np.searchsorted(node_r_scaled, y_limits[0], side="left") - 1)
    i1 = min(len(node_r_scaled), np.searchsorted(node_r_scaled, y_limits[1], side="right") + 1)

    node_z_scaled = node_z_scaled[j0:j1]
    node_r_scaled = node_r_scaled[i0:i1]

    # x-axis: Z, y-axis: R
    z_grid, r_grid = np.meshgrid(node_z_scaled, node_r_scaled)
    density_raw = mesh.n_e_hat[i0:i1, j0:j1]

    if args.log_density:
        density_plot = np.log10(np.clip(density_raw, 1.0e-12, None))
    else:
        density_plot = density_raw

    fig, ax = plt.subplots(figsize=args.figsize_value)
    ax.set_aspect("equal", adjustable="box")

    # Density colormesh: Z on x-axis, R on y-axis
    # Grayscale: high density = black, low density = white
    mesh_plot = ax.pcolormesh(
        z_grid,
        r_grid,
        density_plot,
        shading="gouraud",
        cmap="gray_r",
        vmin=density_vmin,
        vmax=density_vmax,
        zorder=1,
    )

    n_min = float(np.nanmin(density_raw))
    n_max = float(np.nanmax(density_raw))
    if n_min <= 1.0 <= n_max:
        contour = ax.contour(
            z_grid,
            r_grid,
            density_raw,
            levels=[1.0],
            colors="red",
            linestyles="--",
            linewidths=1.1,
            zorder=3,
        )
        ax.clabel(
            contour,
            contour.levels,
            fmt=lambda _: r"$n_{\mathrm{crit}}$",
            inline=True,
            fontsize=8,
            colors="red",
        )

    # Ray trajectories
    traj = snapshot.trajectory
    ray_indices = _subsample_indices(traj.n_rays, args.max_rays)

    segments_list: list[np.ndarray] = []
    power_norm_list: list[np.ndarray] = []
    for ray_idx in ray_indices:
        start = int(traj.offsets[ray_idx])
        end = int(traj.offsets[ray_idx + 1])
        if end - start < 2:
            continue

        z_ray = traj.pos_z_cm[start:end] * unit_scale
        r_ray = traj.pos_r_cm[start:end] * unit_scale
        p_ray = traj.power_erg_s[start:end]

        points = np.column_stack((z_ray, r_ray))
        segments = np.stack((points[:-1], points[1:]), axis=1)
        p0 = float(p_ray[0])
        if p0 <= 0.0:
            p0 = float(np.max(p_ray))
        if p0 <= 0.0:
            p_norm = np.zeros(p_ray.size - 1, dtype=np.float64)
        else:
            p_norm = np.clip(p_ray[:-1] / p0, 0.0, 1.0)
        segments_list.append(segments)
        power_norm_list.append(p_norm)

    ray_lines_obj = None
    if segments_list:
        segments_all = np.concatenate(segments_list, axis=0)
        p_norm_all = np.concatenate(power_norm_list, axis=0)
        ray_lines_obj = LineCollection(
            segments_all,
            cmap="viridis",
            norm=Normalize(vmin=0.0, vmax=1.0),
            linewidths=2.0,
            alpha=args.ray_alpha,
            zorder=5,
        )
        ray_lines_obj.set_array(p_norm_all)
        ax.add_collection(ray_lines_obj)

    ax.set_xlabel(f"Z [{unit_label}]")
    ax.set_ylabel(f"R [{unit_label}]")
    if args.title is None:
        title = f"{snapshot.name} ({snapshot.geometry}), t = {snapshot.time_s:.3e} s"
    else:
        title = args.title
    ax.set_title(title)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)

    # Colorbars anchored to axes height via make_axes_locatable
    divider = make_axes_locatable(ax)
    cax_left = divider.append_axes("left", size="4%", pad=0.6)
    cbar_density = fig.colorbar(mesh_plot, cax=cax_left)
    cax_left.yaxis.set_ticks_position("left")
    cax_left.yaxis.set_label_position("left")
    cbar_label = (
        r"$\log_{10}(n_e / n_{\mathrm{crit}})$"
        if args.log_density
        else r"$n_e / n_{\mathrm{crit}}$"
    )
    cbar_density.set_label(cbar_label)

    if ray_lines_obj is not None:
        cax_right = divider.append_axes("right", size="4%", pad=0.2)
        cbar_ray = fig.colorbar(ray_lines_obj, cax=cax_right)
        cbar_ray.set_label("P / P0")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def _add_bool_option(
    parser: argparse.ArgumentParser,
    flag: str,
    default: bool,
    help_text: str,
) -> None:
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(
            flag,
            action=argparse.BooleanOptionalAction,
            default=default,
            help=help_text,
        )
        return

    name = flag.lstrip("-").replace("-", "_")
    parser.add_argument(flag, dest=name, action="store_true", default=default, help=help_text)
    parser.add_argument(f"--no-{name.replace('_', '-')}", dest=name, action="store_false")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot laser ray trajectories over electron density maps "
            "(n_e/n_crit) from TENRYU snapshot HDF5 files."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input snapshot HDF5 file(s) or output directory containing results/",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output image path. For multiple inputs, provide a directory. "
            "Default is <input_stem>_laser_rays.pdf."
        ),
    )
    parser.add_argument("--cmap", type=str, default="viridis", help="Density colormap.")
    parser.add_argument(
        "--max-rays",
        type=int,
        default=50,
        help="Maximum number of rays to overlay (subsampled if needed).",
    )
    parser.add_argument("--ray-alpha", type=float, default=0.9, help="Ray line alpha.")
    _add_bool_option(
        parser,
        "--log-density",
        default=True,
        help_text="Plot density as log10(n_e/n_crit). Use --no-log-density for linear scale.",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help=(
            "Density color scale minimum in plotted units "
            "(log10 units when --log-density, linear otherwise)."
        ),
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help=(
            "Density color scale maximum in plotted units "
            "(log10 units when --log-density, linear otherwise)."
        ),
    )
    parser.add_argument("--dpi", type=int, default=150, help="Output image DPI.")
    parser.add_argument(
        "--zoom",
        type=str,
        default="auto",
        help=(
            "Plot window: 'auto' (ray Z bounds + adaptive R span), 'full' (full mesh), "
            "or manual 'Rmin,Rmax,Zmin,Zmax' in displayed axis units."
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom plot title (default: auto-generated from snapshot metadata).",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="10,6",
        help="Figure size as width,height in inches (default: 10,6).",
    )
    _add_bool_option(
        parser,
        "--um",
        default=True,
        help_text="Display axes in micrometers. Use --no-um for cm.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Reference HDF5 snapshot for auto-zoom range (e.g. initial snapshot _0000.h5).",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help=(
            "Comma-separated snapshot indices to process when input is a directory "
            "(e.g. --steps 0,5,10). Default: all snapshots."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_deps()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.max_rays < 0:
        parser.error("--max-rays must be >= 0")
    if not (0.0 <= args.ray_alpha <= 1.0):
        parser.error("--ray-alpha must be in [0, 1]")
    if args.dpi <= 0:
        parser.error("--dpi must be > 0")
    try:
        args.figsize_value = _parse_figsize(args.figsize)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.zoom_mode, args.zoom_bounds = _parse_zoom(args.zoom)
    except ValueError as exc:
        parser.error(str(exc))

    step_filter: set[int] | None = None
    if args.steps is not None:
        try:
            step_filter = {int(s.strip()) for s in args.steps.split(",")}
        except ValueError:
            parser.error("--steps must be comma-separated integers")

    input_paths: list[Path] = []
    input_base_dirs: list[Path | None] = []
    base_dir: Path | None = None
    for inp in args.inputs:
        if inp.is_dir():
            base_dir = inp
            try:
                snapshots = _collect_snapshots_from_dir(inp)
            except FileNotFoundError as exc:
                parser.error(str(exc))
            if step_filter is not None:
                snapshots = [
                    p for p in snapshots
                    if int(SNAPSHOT_INDEX_RE.search(p.name).group(1)) in step_filter
                ]
                if not snapshots:
                    parser.error(f"no snapshots matched --steps {args.steps}")
            input_paths.extend(snapshots)
            input_base_dirs.extend([inp] * len(snapshots))
        else:
            input_paths.append(inp)
            input_base_dirs.append(None)
    try:
        if args.output is None and any(path_base_dir != base_dir for path_base_dir in input_base_dirs):
            output_paths = [
                _default_output_path(path, path_base_dir)
                for path, path_base_dir in zip(input_paths, input_base_dirs)
            ]
        else:
            output_paths = _resolve_output_paths(input_paths, args.output, base_dir)
    except ValueError as exc:
        parser.error(str(exc))

    mesh_range_cm = _find_domain_range_cm(input_paths[0], args.reference)
    if mesh_range_cm is not None:
        r_min_cm, r_max_cm = mesh_range_cm
        source = "reference" if args.reference else "frozen_config/mesh"
        print(f"[info] auto-zoom range: r_min = {r_min_cm * 1e4:.1f} um, r_max = {r_max_cm * 1e4:.1f} um ({source})")

    status = 0
    for in_path, out_path in zip(input_paths, output_paths):
        try:
            snapshot = _read_snapshot(in_path)
            _plot_snapshot(snapshot, args, out_path, mesh_range_cm)
            print(f"[ok] {in_path} -> {out_path}")
        except (OSError, SnapshotFormatError, RuntimeError) as exc:
            status = 1
            print(f"[error] {in_path}: {exc}", file=sys.stderr)

    return status


if __name__ == "__main__":
    raise SystemExit(main())
