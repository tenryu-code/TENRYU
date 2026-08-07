#!/usr/bin/env python3
"""Plot laser deposited energy and power time evolution from TENRYU snapshot HDF5 files.

Deposited power is derived from the cumulative energy integral (trapezoidal)
to avoid per-step sampling noise in the raw deposited_power field.
"""

from __future__ import annotations

import argparse
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
        raise RuntimeError("h5py is required. Install: pip install h5py")
    if plt is None:
        raise RuntimeError("matplotlib is required. Install: pip install matplotlib")


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


def _read_scalar_dataset(h5, dataset_path: str) -> float:
    obj = h5.get(dataset_path)
    if obj is None:
        raise SnapshotFormatError(f"missing required dataset '{dataset_path}'")
    arr = np.asarray(obj[...], dtype=np.float64)
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
        return float(raw.reshape(-1)[0])
    raise SnapshotFormatError("missing required time: expected 'time_state/t'")


def _load_laser_data(
    paths: list[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load time, total deposited power, and absorption fraction from snapshots."""
    times: list[float] = []
    total_power: list[float] = []
    abs_frac: list[float] = []

    for path in paths:
        with h5py.File(path, "r") as h5:
            t = _read_snapshot_time_s(h5)

            dep = h5.get("laser/deposited_power")
            vol = h5.get("hydro/vol")
            if dep is not None and vol is not None:
                dep_arr = np.asarray(dep[...], dtype=np.float64)
                vol_arr = np.asarray(vol[...], dtype=np.float64)
                p = float(np.sum(dep_arr * vol_arr))
            elif dep is not None:
                p = float(np.sum(np.asarray(dep[...], dtype=np.float64)))
            else:
                p = 0.0

            af_ds = h5.get("laser/absorption_fraction")
            if af_ds is not None:
                af = float(np.asarray(af_ds[...], dtype=np.float64).reshape(-1)[0])
            else:
                af = 0.0

        times.append(t)
        total_power.append(p)
        abs_frac.append(af)

    return np.array(times), np.array(total_power), np.array(abs_frac)


def _compute_cumulative_energy(times: np.ndarray, power: np.ndarray) -> np.ndarray:
    """Integrate power over time using trapezoidal rule to get cumulative energy."""
    energy = np.zeros_like(times)
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        energy[i] = energy[i - 1] + 0.5 * (power[i - 1] + power[i]) * dt
    return energy


def _derivative_from_cumulative(
    times: np.ndarray, energy: np.ndarray, window: int = 3
) -> np.ndarray:
    """Compute smoothed power from cumulative energy using central differences.

    Uses a symmetric window of ±window snapshots for the finite difference,
    which naturally smooths out per-step sampling noise.
    """
    n = len(times)
    power = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n - 1, i + window)
        if hi == lo:
            power[i] = 0.0
        else:
            power[i] = (energy[hi] - energy[lo]) / (times[hi] - times[lo])
    return power


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot laser deposited energy/power time evolution from TENRYU snapshots."
    )
    parser.add_argument(
        "snapshot_dir",
        type=Path,
        help="Directory containing *_NNNN.h5 files, or a TENRYU output directory with results/",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: <snapshot_dir>/figs/<run_name>_laser_energy.png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI (default: 150)",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[12.0, 8.0],
        metavar=("W", "H"),
        help="Figure size in inches (default: 12 8)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=3,
        help="Half-window size for power smoothing (default: 3 snapshots)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_deps()
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        snapshot_paths = _collect_snapshot_paths(args.snapshot_dir)
        run_name = _infer_run_name(snapshot_paths)
        print(f"[info] loading {len(snapshot_paths)} snapshots ...")
        times, power_raw, abs_frac = _load_laser_data(snapshot_paths)
    except (OSError, SnapshotFormatError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    if len(times) < 2:
        print("[error] need at least 2 snapshots to plot", file=sys.stderr)
        return 1

    # Compute cumulative energy from raw snapshot power
    energy = _compute_cumulative_energy(times, power_raw)

    # Derive smoothed power from cumulative energy (robust to per-step noise)
    power_smooth = _derivative_from_cumulative(times, energy, window=args.smooth_window)

    times_ns = times * 1.0e9
    energy_J = energy * 1.0e-7  # erg -> J
    power_raw_TW = power_raw * 1.0e-19
    power_smooth_TW = power_smooth * 1.0e-19

    # Determine output path
    if args.output is not None:
        output_path = args.output
    else:
        figs_dir = args.snapshot_dir / "figs"
        output_path = figs_dir / f"{run_name}_laser_energy.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=tuple(args.figsize), sharex=True)
    fig.suptitle(f"{run_name} — Laser Energy Deposition", fontsize=14)

    # Panel 1: Cumulative deposited energy
    ax1 = axes[0]
    ax1.plot(times_ns, energy_J, "b-", linewidth=1.5)
    ax1.set_ylabel("Cumulative Deposited\nEnergy [J]")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Deposited power (raw + smoothed)
    ax2 = axes[1]
    ax2.plot(times_ns, power_raw_TW, color="salmon", linewidth=0.5, alpha=0.5,
             label="Raw (per-snapshot)")
    ax2.plot(times_ns, power_smooth_TW, "r-", linewidth=1.5,
             label=f"Smoothed (±{args.smooth_window} snapshots)")
    ax2.set_ylabel("Deposited Power\n[TW]")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Absorption fraction
    ax3 = axes[2]
    ax3.plot(times_ns, abs_frac * 100.0, "g-", linewidth=1.0)
    ax3.set_ylabel("Absorption\nFraction [%]")
    ax3.set_xlabel("Time [ns]")
    ax3.set_ylim(-5, 105)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()

    print(f"[info] writing {output_path} ...")
    fig.savefig(str(output_path), dpi=args.dpi, bbox_inches="tight")
    print(f"[ok] wrote {output_path}")

    # Print summary
    print(f"\n[summary]")
    print(f"  Time range: {times_ns[0]:.3f} - {times_ns[-1]:.3f} ns")
    print(f"  Total deposited energy: {energy_J[-1]:.4f} J")
    print(f"  Peak deposited power (smoothed): {np.max(power_smooth_TW):.4f} TW")
    print(f"  Mean absorption fraction: {np.mean(abs_frac)*100:.1f}%")

    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
