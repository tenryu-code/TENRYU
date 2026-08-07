#!/usr/bin/env python3
"""Generate a 2D RZ density animation from TENRYU snapshot HDF5 files."""

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
    from matplotlib.animation import FuncAnimation
except ModuleNotFoundError:
    plt = None
    FuncAnimation = None


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
    name = paths[0].stem
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


def _read_mesh_and_rho(
    h5, path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_r = np.asarray(_require_dataset(h5, "mesh/x_r")[...], dtype=np.float64)
    x_z = np.asarray(_require_dataset(h5, "mesh/x_z")[...], dtype=np.float64)
    rho = np.asarray(_require_dataset(h5, "hydro/rho")[...], dtype=np.float64)

    if x_r.ndim != 2:
        raise SnapshotFormatError(f"dataset 'mesh/x_r' must be rank-2, got rank {x_r.ndim}")
    if x_z.ndim != 2:
        raise SnapshotFormatError(f"dataset 'mesh/x_z' must be rank-2, got rank {x_z.ndim}")
    if x_r.shape != x_z.shape:
        raise SnapshotFormatError(
            f"mesh coordinate shape mismatch: mesh/x_r has shape {x_r.shape}, "
            f"mesh/x_z has shape {x_z.shape}"
        )
    if x_r.shape[0] < 2 or x_r.shape[1] < 2:
        raise SnapshotFormatError(
            f"mesh coordinate dimensions must both be at least 2, got shape {x_r.shape}"
        )
    if rho.ndim != 1:
        raise SnapshotFormatError(f"dataset 'hydro/rho' must be rank-1, got rank {rho.ndim}")

    nr = x_r.shape[0] - 1
    nz = x_r.shape[1] - 1
    expected_cells = nr * nz
    if rho.size != expected_cells:
        raise SnapshotFormatError(
            f"cell count mismatch in {path.name}: hydro/rho has {rho.size}, "
            f"mesh implies {expected_cells}"
        )
    return x_r, x_z, rho.reshape(nr, nz)


def _load_all_snapshots(
    paths: list[Path],
) -> tuple[
    list[float],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]:
    times_s: list[float] = []
    all_xr: list[np.ndarray] = []
    all_xz: list[np.ndarray] = []
    all_rho2: list[np.ndarray] = []

    for path in paths:
        with h5py.File(path, "r") as h5:
            t = _read_snapshot_time_s(h5)
            x_r, x_z, rho2 = _read_mesh_and_rho(h5, path)
        times_s.append(t)
        all_xr.append(x_r)
        all_xz.append(x_z)
        all_rho2.append(rho2)

    return times_s, all_xr, all_xz, all_rho2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a 2D RZ density animation from TENRYU snapshot files."
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
        help="Output base path; both .mp4 and .gif are generated (default: <snapshot_dir>/figs/<run_name>_density2d_anim)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second (default: 10)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI (default: 150)",
    )
    parser.add_argument(
        "--rho-min",
        type=float,
        default=None,
        help="Min density for color scale in g/cm^3 (auto if omitted)",
    )
    parser.add_argument(
        "--rho-max",
        type=float,
        default=None,
        help="Max density for color scale in g/cm^3 (auto if omitted)",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        default=False,
        help="Use log scale for density colors",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="inferno",
        help="Matplotlib colormap (default: inferno)",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        default=False,
        help="Disable mirroring across r=0",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[10.0, 6.0],
        metavar=("W", "H"),
        help="Figure size in inches (default: 10 6)",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated step indices or ranges to include (e.g. '0,5-10,20')",
    )
    return parser


def _parse_steps(steps_str: str, n_total: int) -> list[int]:
    indices: set[int] = set()
    for part in steps_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo_i = int(lo)
            hi_i = int(hi)
            for i in range(lo_i, min(hi_i + 1, n_total)):
                if 0 <= i < n_total:
                    indices.add(i)
        else:
            i = int(part)
            if 0 <= i < n_total:
                indices.add(i)
    return sorted(indices)


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_deps()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.fps <= 0:
        parser.error("--fps must be > 0")
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

        if args.steps is not None:
            selected = _parse_steps(args.steps, len(snapshot_paths))
            if not selected:
                print("[error] --steps yielded no valid indices", file=sys.stderr)
                return 1
            snapshot_paths = [snapshot_paths[i] for i in selected]

        print(f"[info] loading {len(snapshot_paths)} snapshots ...")
        times_s, all_xr, all_xz, all_rho2 = _load_all_snapshots(snapshot_paths)
    except (OSError, SnapshotFormatError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    times_ns = [t * 1.0e9 for t in times_s]

    global_rho_min = min(float(np.min(rho2)) for rho2 in all_rho2)
    global_rho_max = max(float(np.max(rho2)) for rho2 in all_rho2)
    vmin = args.rho_min if args.rho_min is not None else global_rho_min
    vmax = args.rho_max if args.rho_max is not None else global_rho_max

    if args.log:
        positive_rho = [rho2[rho2 > 0.0] for rho2 in all_rho2]
        positive_rho = [values for values in positive_rho if values.size > 0]
        if not positive_rho:
            parser.error("--log requires at least one positive density")
        if vmin <= 0.0:
            vmin = min(float(np.min(values)) for values in positive_rho)
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    zmin_cm = min(float(np.min(x_z)) for x_z in all_xz)
    zmax_cm = max(float(np.max(x_z)) for x_z in all_xz)
    rmax_cm = max(float(np.max(x_r)) for x_r in all_xr)
    xmin_um = zmin_cm * 1.0e4
    xmax_um = zmax_cm * 1.0e4
    rlimit_um = rmax_cm * 1.02 * 1.0e4
    if args.no_mirror:
        ymin_um = 0.0
        ymax_um = rlimit_um
    else:
        ymin_um = -rlimit_um
        ymax_um = rlimit_um

    figs_dir = args.snapshot_dir / "figs"
    base_stem = f"{run_name}_density2d_anim"
    if args.output is not None:
        output_mp4 = args.output.with_suffix(".mp4")
        output_gif = args.output.with_suffix(".gif")
    else:
        output_mp4 = figs_dir / f"{base_stem}.mp4"
        output_gif = figs_dir / f"{base_stem}.gif"

    # Build animation
    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=args.cmap)
    colorbar = fig.colorbar(scalar_mappable, ax=ax)
    colorbar.set_label("ρ [g/cm³]")

    def update(frame_idx):
        ax.clear()
        r_um = all_xr[frame_idx] * 1.0e4
        z_um = all_xz[frame_idx] * 1.0e4
        rho2 = all_rho2[frame_idx]

        meshes = [
            ax.pcolormesh(
                z_um, r_um, rho2, shading="flat", cmap=args.cmap, norm=norm
            )
        ]
        if not args.no_mirror:
            meshes.append(
                ax.pcolormesh(
                    z_um, -r_um, rho2, shading="flat", cmap=args.cmap, norm=norm
                )
            )

        ax.set_xlim(xmin_um, xmax_um)
        ax.set_ylim(ymin_um, ymax_um)
        ax.set_aspect("equal")
        ax.set_xlabel("z [μm]")
        ax.set_ylabel("r [μm]")
        ax.set_title(f"{run_name} — Density (2D RZ)")
        ax.grid(True, alpha=0.3)
        time_text = ax.text(
            0.98, 0.95, f"t = {times_ns[frame_idx]:.3f} ns", transform=ax.transAxes,
            fontsize=12, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
        )
        return (*meshes, time_text)

    n_frames = len(times_s)
    anim = FuncAnimation(
        fig, update, frames=n_frames,
        blit=False, interval=1000 // args.fps,
    )

    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    # Save MP4 (ffmpeg)
    try:
        print(f"[info] writing {n_frames} frames to {output_mp4} (writer=ffmpeg) ...")
        anim.save(str(output_mp4), writer="ffmpeg", fps=args.fps, dpi=args.dpi)
        print(f"[ok] wrote {output_mp4}")
    except Exception as exc:
        print(f"[warn] ffmpeg failed ({exc}), skipping MP4", file=sys.stderr)

    # Save GIF (pillow)
    try:
        print(f"[info] writing {n_frames} frames to {output_gif} (writer=pillow) ...")
        anim.save(str(output_gif), writer="pillow", fps=args.fps, dpi=args.dpi)
        print(f"[ok] wrote {output_gif}")
    except Exception as exc:
        print(f"[error] GIF save failed: {exc}", file=sys.stderr)

    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
