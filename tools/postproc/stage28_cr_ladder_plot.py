#!/usr/bin/env python3
"""Plot CR ladder diagnostics from a history HDF5 file.

Usage: python tools/postproc/stage28_cr_ladder_plot.py --input history.h5 --out-dir tmp/stage28-plots/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ICF_PATH = "/diagnostics/icf/v1"
CONSERVATION_PATH = "/diagnostics/conservation/v1"


def finite_argmax(values: np.ndarray) -> int | None:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return None
    indices = np.flatnonzero(finite)
    return int(indices[np.argmax(arr[finite])])


def finite_argmin(values: np.ndarray) -> int | None:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return None
    indices = np.flatnonzero(finite)
    return int(indices[np.argmin(arr[finite])])


def read_1d(handle: h5py.File, path: str) -> np.ndarray | None:
    if path not in handle or not isinstance(handle[path], h5py.Dataset):
        return None
    arr = np.asarray(handle[path][()])
    if arr.ndim != 1:
        return None
    return arr.astype(float, copy=False)


def first_available(handle: h5py.File, paths: tuple[str, ...]) -> np.ndarray | None:
    for path in paths:
        arr = read_1d(handle, path)
        if arr is not None:
            return arr
    return None


def align_xy(time: np.ndarray | None, values: np.ndarray | None) -> tuple[np.ndarray, np.ndarray] | None:
    if values is None or values.size == 0:
        return None
    y = np.asarray(values, dtype=float).reshape(-1)
    if time is None or time.size == 0:
        x = np.arange(y.size, dtype=float)
    else:
        n = min(time.size, y.size)
        x = np.asarray(time[:n], dtype=float)
        y = y[:n]
    return x, y


def value_at(values: np.ndarray | None, index: int | None) -> float | None:
    if values is None or index is None or index < 0 or index >= values.size:
        return None
    return float(values[index])


def format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(fvalue):
        return "nan"
    return f"{fvalue:.6e}"


def stem_without_h5(path: Path) -> str:
    if path.name.endswith(".h5"):
        return path.name[:-3]
    if path.name.endswith(".hdf5"):
        return path.name[:-5]
    return path.stem


def load_data(path: Path) -> dict[str, Any]:
    try:
        with h5py.File(path, "r") as handle:
            data: dict[str, Any] = {
                "time": first_available(handle, ("/history/time", f"{ICF_PATH}/time_s", f"{CONSERVATION_PATH}/time_s")),
                "step": first_available(handle, ("/history/step", f"{ICF_PATH}/step", f"{CONSERVATION_PATH}/step")),
                "CR": read_1d(handle, f"{ICF_PATH}/CR"),
                "shell_radius_cm": read_1d(handle, f"{ICF_PATH}/shell_radius_cm"),
                "IFAR": read_1d(handle, f"{ICF_PATH}/IFAR"),
                "eps_E": {},
            }
            if CONSERVATION_PATH in handle and isinstance(handle[CONSERVATION_PATH], h5py.Group):
                for name, obj in handle[CONSERVATION_PATH].items():
                    if name.startswith("eps_E_") and isinstance(obj, h5py.Dataset):
                        arr = np.asarray(obj[()])
                        if arr.ndim == 1:
                            data["eps_E"][name] = arr.astype(float, copy=False)
            return data
    except OSError as exc:
        print(f"error: failed to read HDF5 file {path}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def compute_metrics(data: dict[str, Any]) -> dict[str, Any]:
    time = data["time"]
    step = data["step"]
    cr = data["CR"]
    radius = data["shell_radius_cm"]
    cr_idx = finite_argmax(cr) if cr is not None else None
    radius_idx = finite_argmin(radius) if radius is not None else None
    final_cr = None
    if cr is not None and cr.size > 0:
        finite = cr[np.isfinite(cr)]
        final_cr = None if finite.size == 0 else float(finite[-1])
    return {
        "CR_max": value_at(cr, cr_idx),
        "t_CR_max_s": value_at(time, cr_idx),
        "step_CR_max": value_at(step, cr_idx),
        "final_CR": final_cr,
        "R_shell_min_cm": value_at(radius, radius_idx),
        "t_R_shell_min_s": value_at(time, radius_idx),
    }


def plot_log(ax: plt.Axes, x: np.ndarray, y: np.ndarray, label: str | None = None) -> None:
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
    if np.any(mask):
        ax.semilogy(x[mask], y[mask], label=label)


def make_plot(input_path: Path, out_dir: Path, data: dict[str, Any], metrics: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem_without_h5(input_path)}_cr_ladder.png"
    time = data["time"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    aligned = align_xy(time, data["CR"])
    if aligned is not None:
        x, y = aligned
        plot_log(axes[0, 0], x, y)
        if metrics["CR_max"] is not None and metrics["t_CR_max_s"] is not None:
            axes[0, 0].annotate(
                f"CR_max={metrics['CR_max']:.3e}",
                xy=(metrics["t_CR_max_s"], metrics["CR_max"]),
                xytext=(0.05, 0.9),
                textcoords="axes fraction",
                arrowprops={"arrowstyle": "->", "linewidth": 0.8},
            )
    axes[0, 0].set_title("CR vs time")
    axes[0, 0].set_xlabel("time [s]")
    axes[0, 0].set_ylabel("CR")
    axes[0, 0].grid(True, which="both", alpha=0.3)

    aligned = align_xy(time, data["shell_radius_cm"])
    if aligned is not None:
        x, y = aligned
        plot_log(axes[0, 1], x, y)
    axes[0, 1].set_title("shell_radius_cm vs time")
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].set_ylabel("shell_radius_cm [cm]")
    axes[0, 1].grid(True, which="both", alpha=0.3)

    aligned = align_xy(time, data["IFAR"])
    if aligned is not None:
        x, y = aligned
        mask = np.isfinite(x) & np.isfinite(y)
        axes[1, 0].plot(x[mask], y[mask])
    axes[1, 0].set_title("IFAR vs time")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("IFAR")
    axes[1, 0].grid(True, alpha=0.3)

    for name, values in sorted(data["eps_E"].items()):
        aligned = align_xy(time, values)
        if aligned is None:
            continue
        x, y = aligned
        plot_log(axes[1, 1], x, np.abs(y), label=name.removeprefix("eps_E_"))
    axes[1, 1].set_title("operator energy residuals")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("|eps_E|")
    axes[1, 1].grid(True, which="both", alpha=0.3)
    if data["eps_E"]:
        axes[1, 1].legend(fontsize="small")

    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("tmp/stage28-plots/"), type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    data = load_data(args.input)
    metrics = compute_metrics(data)
    plot_path = make_plot(args.input, args.out_dir, data, metrics)
    metrics["plot_path"] = str(plot_path)

    print(f"## CR ladder summary — {args.input.name}")
    print()
    print("| Quantity | Value |")
    print("|----------|-------|")
    print(f"| CR_max | {format_value(metrics['CR_max'])} |")
    print(f"| t at CR_max [s] | {format_value(metrics['t_CR_max_s'])} |")
    print(f"| Final CR | {format_value(metrics['final_CR'])} |")
    print(f"| R_shell_min [cm] | {format_value(metrics['R_shell_min_cm'])} |")
    print(f"| t at R_shell_min [s] | {format_value(metrics['t_R_shell_min_s'])} |")
    print(f"| Plot saved to | {plot_path} |")

    if args.json_out is not None:
        write_json(args.json_out, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
