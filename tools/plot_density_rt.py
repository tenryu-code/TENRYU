#!/usr/bin/env python3
"""Plot an r-t density colormap from TENRYU snapshot HDF5 files."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import h5py
except ModuleNotFoundError:
    h5py = None

try:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


SNAPSHOT_GLOB = "*_[0-9][0-9][0-9][0-9].h5"
SNAPSHOT_INDEX_RE = re.compile(r"_(\d{4})\.h5$")


class SnapshotFormatError(RuntimeError):
    """Raised when snapshot contents do not match expected TENRYU layout."""


def _ensure_deps() -> None:
    if h5py is None:
        raise RuntimeError("h5py is required. Install dependency: pip install h5py")
    if plt is None:
        raise RuntimeError("matplotlib is required. Install dependency: pip install matplotlib")


def _snapshot_index(path: Path) -> int:
    match = SNAPSHOT_INDEX_RE.search(path.name)
    if match is None:
        raise SnapshotFormatError(f"snapshot file does not end with _NNNN.h5: {path}")
    return int(match.group(1))


def _collect_snapshot_paths(snapshot_dir: Path) -> list[Path]:
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"snapshot directory not found: {snapshot_dir}")
    if not snapshot_dir.is_dir():
        raise NotADirectoryError(f"snapshot_dir is not a directory: {snapshot_dir}")

    results_dir = snapshot_dir / "results"
    scan_dir = results_dir if results_dir.is_dir() else snapshot_dir

    paths: list[Path] = []
    for path in scan_dir.glob(SNAPSHOT_GLOB):
        if not path.is_file():
            continue
        name = path.name
        if "_ckpt_" in name or "_history" in name:
            continue
        if SNAPSHOT_INDEX_RE.search(name) is None:
            continue
        paths.append(path)

    paths.sort(key=lambda p: (_snapshot_index(p), p.name))
    if not paths:
        raise FileNotFoundError(
            f"no snapshot files matched '{SNAPSHOT_GLOB}' in {scan_dir}"
        )
    return paths


def _infer_run_name(paths: list[Path]) -> str:
    """Extract run name from snapshot filenames (e.g. 'gxii_shell_1D_0000.h5' -> 'gxii_shell_1D')."""
    name = paths[0].stem  # e.g. 'gxii_shell_1D_0000'
    m = re.match(r"^(.+)_\d{4}$", name)
    return m.group(1) if m else name


def _require_dataset(h5, dataset_path: str):
    obj = h5.get(dataset_path)
    if obj is None:
        raise SnapshotFormatError(f"missing required dataset '{dataset_path}'")
    if not isinstance(obj, h5py.Dataset):
        raise SnapshotFormatError(f"expected dataset at '{dataset_path}', got {type(obj).__name__}")
    return obj


def _read_scalar_dataset(h5, dataset_path: str) -> float:
    ds = _require_dataset(h5, dataset_path)
    arr = np.asarray(ds[...], dtype=np.float64)
    if arr.shape not in ((), (1,)):
        raise SnapshotFormatError(
            f"dataset '{dataset_path}' must be scalar, got shape {arr.shape}"
        )
    return float(arr.reshape(-1)[0])


def _read_snapshot_time_s(h5) -> float:
    if "time_state/t" in h5:
        return _read_scalar_dataset(h5, "time_state/t")

    if "t" in h5.attrs:
        raw = np.asarray(h5.attrs["t"], dtype=np.float64)
        if raw.shape not in ((), (1,)):
            raise SnapshotFormatError(
                f"root attribute 't' must be scalar, got shape {raw.shape}"
            )
        return float(raw.reshape(-1)[0])

    raise SnapshotFormatError("missing required time: expected 'time_state/t' (or root attr 't')")


def _read_centers_and_rho(h5, path: Path) -> tuple[np.ndarray, np.ndarray]:
    node_r = np.asarray(_require_dataset(h5, "mesh/x_r")[...], dtype=np.float64)
    rho = np.asarray(_require_dataset(h5, "hydro/rho")[...], dtype=np.float64)

    if node_r.ndim != 1:
        raise SnapshotFormatError(f"dataset 'mesh/x_r' must be rank-1, got rank {node_r.ndim}")
    if node_r.size < 2:
        raise SnapshotFormatError(
            f"dataset 'mesh/x_r' must contain at least 2 nodes, got {node_r.size}"
        )
    if rho.ndim != 1:
        raise SnapshotFormatError(f"dataset 'hydro/rho' must be rank-1, got rank {rho.ndim}")

    centers = 0.5 * (node_r[:-1] + node_r[1:])
    if rho.size != centers.size:
        raise SnapshotFormatError(
            f"cell count mismatch in {path.name}: hydro/rho has {rho.size}, mesh implies {centers.size}"
        )
    if np.any(np.diff(centers) <= 0.0):
        raise SnapshotFormatError(f"cell centers are not strictly increasing in {path.name}")
    return centers, rho


def _scan_radial_extent(paths: list[Path]) -> tuple[float, float]:
    r_min = np.inf
    r_max = -np.inf
    for path in paths:
        with h5py.File(path, "r") as h5:
            centers, _ = _read_centers_and_rho(h5, path)
        r_min = min(r_min, float(np.min(centers)))
        r_max = max(r_max, float(np.max(centers)))

    if not np.isfinite(r_min) or not np.isfinite(r_max) or r_min >= r_max:
        raise SnapshotFormatError(
            f"invalid radial extent across snapshots: r_min={r_min}, r_max={r_max}"
        )
    return r_min, r_max


def _get_radial_extent(paths: list[Path]) -> tuple[float, float]:
    first_path = paths[0]
    with h5py.File(first_path, "r") as h5:
        centers, _ = _read_centers_and_rho(h5, first_path)
        r_min = float(np.min(centers))
        frozen_obj = h5.get("metadata/frozen_config")

        frozen_r_max: float | None = None
        if isinstance(frozen_obj, h5py.Dataset):
            raw = frozen_obj[()]
            if isinstance(raw, np.ndarray):
                if raw.shape in ((), (1,)):
                    raw = raw.reshape(-1)[0]
                else:
                    raw = None

            text: str | None = None
            if isinstance(raw, (bytes, np.bytes_)):
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    text = None
            elif isinstance(raw, str):
                text = raw

            if text is not None:
                try:
                    frozen_config = json.loads(text)
                except json.JSONDecodeError:
                    frozen_config = None

                if isinstance(frozen_config, dict):
                    mesh = frozen_config.get("mesh")
                    if isinstance(mesh, dict) and "r_max" in mesh:
                        try:
                            frozen_r_max = float(mesh["r_max"])
                        except (TypeError, ValueError):
                            frozen_r_max = None

        if frozen_r_max is not None:
            if not np.isfinite(r_min) or not np.isfinite(frozen_r_max) or r_min >= frozen_r_max:
                raise SnapshotFormatError(
                    f"invalid radial extent from frozen_config in {first_path.name}: "
                    f"r_min={r_min}, r_max={frozen_r_max}"
                )
            return r_min, frozen_r_max

    return _scan_radial_extent(paths)


def _pass2_interpolate_profiles(
    paths: list[Path],
    r_grid_cm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_snap = len(paths)
    times_s = np.empty(n_snap, dtype=np.float64)
    rho_grid = np.empty((n_snap, r_grid_cm.size), dtype=np.float64)

    for i, path in enumerate(paths):
        with h5py.File(path, "r") as h5:
            centers, rho = _read_centers_and_rho(h5, path)
            times_s[i] = _read_snapshot_time_s(h5)
        rho_grid[i, :] = np.interp(r_grid_cm, centers, rho)

    return times_s, rho_grid


def _resolve_limits(
    rho: np.ndarray,
    rho_min: float | None,
    rho_max: float | None,
) -> tuple[float | None, float | None]:
    vmin = rho_min if rho_min is not None else float(np.nanmin(rho))
    vmax = rho_max if rho_max is not None else float(np.nanmax(rho))
    return vmin, vmax


def _cell_edges(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 1:
        raise ValueError(f"plot coordinates must be rank-1, got rank {coords.ndim}")
    if coords.size == 0:
        raise ValueError("plot coordinates must not be empty")
    if coords.size == 1:
        half_width = 0.5 * max(abs(float(coords[0])), 1.0)
        return np.array([coords[0] - half_width, coords[0] + half_width], dtype=np.float64)

    delta = np.diff(coords)
    if np.any(delta <= 0.0):
        raise ValueError("plot coordinates must be strictly increasing")

    edges = np.empty(coords.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (coords[:-1] + coords[1:])
    edges[0] = coords[0] - 0.5 * delta[0]
    edges[-1] = coords[-1] + 0.5 * delta[-1]
    return edges


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an r-t colormap of density from TENRYU snapshot files."
    )
    parser.add_argument(
        "snapshot_dir",
        type=Path,
        help="Directory containing *_NNNN.h5 files, or a TENRYU output directory with a results/ subdirectory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <snapshot_dir>/figs/<run_name>_density_rt.pdf)",
    )
    parser.add_argument(
        "--nr-grid",
        type=int,
        default=500,
        help="Number of radial grid points (default: 500)",
    )
    parser.add_argument(
        "--rho-min",
        type=float,
        default=None,
        help="Min density for colorbar in g/cm^3 (auto if omitted)",
    )
    parser.add_argument(
        "--rho-max",
        type=float,
        default=None,
        help="Max density for colorbar in g/cm^3 (auto if omitted)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI (default: 300)",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="inferno",
        help="Matplotlib colormap name (default: inferno)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_deps()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.nr_grid < 2:
        parser.error("--nr-grid must be >= 2")
    if args.dpi <= 0:
        parser.error("--dpi must be > 0")
    if args.rho_min is not None and args.rho_min <= 0.0:
        parser.error("--rho-min must be > 0")
    if args.rho_max is not None and args.rho_max <= 0.0:
        parser.error("--rho-max must be > 0")
    if args.rho_min is not None and args.rho_max is not None and args.rho_max <= args.rho_min:
        parser.error("--rho-max must be greater than --rho-min")

    try:
        snapshot_paths = _collect_snapshot_paths(args.snapshot_dir)
        run_name = _infer_run_name(snapshot_paths)
        r_min, r_max = _get_radial_extent(snapshot_paths)
        r_grid_cm = np.linspace(r_min, r_max, args.nr_grid, dtype=np.float64)
        times_s, rho_grid = _pass2_interpolate_profiles(snapshot_paths, r_grid_cm)
        time_ns = times_s * 1.0e9
        radius_um = r_grid_cm * 1.0e4
        time_edges_ns = _cell_edges(time_ns)
        radius_edges_um = _cell_edges(radius_um)
    except (OSError, SnapshotFormatError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    output_path = (
        args.output
        if args.output is not None
        else args.snapshot_dir / "figs" / f"{run_name}_density_rt.pdf"
    )

    vmin, vmax = _resolve_limits(rho_grid, args.rho_min, args.rho_max)

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    image = ax.imshow(
        rho_grid.T,
        origin="lower",
        aspect="auto",
        extent=(
            float(time_edges_ns[0]),
            float(time_edges_ns[-1]),
            float(radius_edges_um[0]),
            float(radius_edges_um[-1]),
        ),
        interpolation="bilinear",
        cmap=args.cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Radius [μm]")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("ρ [g/cm³]")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
