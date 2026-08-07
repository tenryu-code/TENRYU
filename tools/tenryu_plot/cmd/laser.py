"""Laser diagnostic plot command."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..io import History, RunDir, Snapshot, pick_snapshots
from ..style import ensure_matplotlib, format_time, save_figure, time_unit


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("laser", help="Plot laser deposition diagnostics.")
    parser.add_argument("run", type=Path, help="TENRYU output directory or results/ directory")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--times", default=None, help="Comma-separated time selectors")
    selector.add_argument("--all-snapshots", action="store_true", default=False, help="Plot all snapshots")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=140, help="Output DPI (default: 140)")
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> int:
    run_dir = RunDir.find(args.run)
    render_laser(
        run_dir,
        out_path=args.out,
        dpi=args.dpi,
        times_arg=args.times,
        all_snapshots=args.all_snapshots,
    )
    return 0


def render_laser(
    run_dir: RunDir,
    out_path: str | Path | None,
    dpi: int,
    times_arg: str | None = None,
    all_snapshots: bool = False,
) -> Path:
    if dpi <= 0:
        raise ValueError("--dpi must be > 0")
    all_loaded = [Snapshot.load(path) for path in run_dir.snapshot_paths]
    history = _load_history(run_dir)
    has_snapshot_laser = any(snapshot.laser_deposited_power is not None for snapshot in all_loaded)
    has_history_laser = history is not None and bool(history.group("laser/"))
    if not has_snapshot_laser and not has_history_laser:
        raise ValueError("run has no laser group in snapshots and no laser/* history")

    selected = _select_snapshots(run_dir, all_loaded, times_arg, all_snapshots)
    nrows = 2 if has_history_laser else 1
    plt = ensure_matplotlib()
    fig, axes_obj = plt.subplots(nrows, 1, figsize=(7.2, 3.4 * nrows), squeeze=False)
    axes = [axes_obj[row][0] for row in range(nrows)]

    _plot_snapshot_deposition(axes[0], selected)
    if has_history_laser and history is not None:
        _plot_history_laser(axes[1], history)

    fig.suptitle(f"{run_dir.case}: laser")
    output = Path(out_path) if out_path is not None else run_dir.path / "plots" / "laser.png"
    saved = save_figure(fig, output, dpi)
    plt.close(fig)
    return saved


def _select_snapshots(
    run_dir: RunDir,
    all_loaded: list[Snapshot],
    times_arg: str | None,
    all_snapshots: bool,
) -> list[Snapshot]:
    if all_snapshots:
        return all_loaded
    if times_arg is not None:
        paths = [pick_snapshots(run_dir, at=part.strip())[0] for part in times_arg.split(",") if part.strip()]
        selected: list[Snapshot] = []
        seen: set[Path] = set()
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            selected.append(Snapshot.load(path))
        return selected

    first = 0
    for index, snapshot in enumerate(all_loaded):
        deposited = snapshot.laser_deposited_power
        if deposited is not None and np.any(np.asarray(deposited, dtype=np.float64) != 0.0):
            first = index
            break
    last = len(all_loaded) - 1
    count = min(3, last - first + 1)
    indices = [int(round(value)) for value in np.linspace(first, last, count)]
    return [all_loaded[index] for index in dict.fromkeys(indices)]


def _plot_snapshot_deposition(ax, snapshots: list[Snapshot]) -> None:
    plotted = False
    for snapshot in snapshots:
        deposited = snapshot.laser_deposited_power
        if deposited is None:
            continue
        values = np.ma.masked_less_equal(np.asarray(deposited, dtype=np.float64), 0.0)
        ax.plot(snapshot.r_centers, values, linewidth=1.1, label=f"t={format_time(snapshot.t)}")
        plotted = True
    ax.set_xlabel("r [cm]")
    ax.set_ylabel("deposited_power [erg/cm^3/s]")
    ax.set_yscale("log")
    ax.set_title("deposition")
    ax.grid(True, which="both", alpha=0.25)
    if plotted:
        ax.legend(fontsize=8)
    else:
        _mark_na(ax)


def _plot_history_laser(ax, history: History) -> None:
    time_scale, time_label = time_unit(history.t[-1] if len(history.t) else 0)
    time_values = history.t * time_scale
    plotted_left = False
    for name in ("laser/absorbed_power_total", "laser/commanded_power_total"):
        arr = history.get(name)
        if arr is None:
            continue
        ax.plot(time_values, np.asarray(arr, dtype=np.float64), linewidth=1.1, label=name.rsplit("/", 1)[-1])
        plotted_left = True
    ax.set_xlabel(f"t [{time_label}]")
    ax.set_ylabel("Power [erg/s]")
    ax.set_title("history")
    ax.grid(True, alpha=0.25)
    if plotted_left:
        ax.legend(fontsize=8, loc="upper left")

    radius_ax = ax.twinx()
    plotted_right = False
    for name, linestyle in (
        ("laser/critical_surface_r", "--"),
        ("laser/absorption_weighted_r", ":"),
    ):
        arr = history.get(name)
        if arr is None:
            continue
        radius_ax.plot(
            time_values,
            np.asarray(arr, dtype=np.float64),
            linewidth=1.1,
            linestyle=linestyle,
            label=name.rsplit("/", 1)[-1],
        )
        plotted_right = True
    radius_ax.set_ylabel("r [cm]")
    if plotted_right:
        radius_ax.legend(fontsize=8, loc="upper right")
    if not plotted_left and not plotted_right:
        _mark_na(ax)


def _load_history(run_dir: RunDir) -> History | None:
    if run_dir.history_path is None:
        return None
    return History.load(run_dir.history_path)


def _mark_na(ax) -> None:
    ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center", va="center")
