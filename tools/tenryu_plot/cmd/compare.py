"""Run-to-run profile comparison command."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from .. import derived
from ..io import History, RunDir, Snapshot, pick_snapshots
from ..style import FIELD_META, axis_label, ensure_matplotlib, format_time, save_figure, time_unit, xunit_scale


DEFAULT_FIELDS = ("rho", "Te", "Ti", "Tr", "u", "P")
TINY = 1.0e-300


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("compare", help="Compare profiles from two runs.")
    parser.add_argument("run_a", type=Path, help="First TENRYU output directory")
    parser.add_argument("run_b", type=Path, help="Second TENRYU output directory")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--at", default=None, help="Time selector, e.g. 1.2ns or 3e-9")
    selector.add_argument("--last", action="store_true", default=False, help="Use the last common time")
    parser.add_argument("--fields", default=",".join(DEFAULT_FIELDS), help="Comma-separated fields")
    parser.add_argument("--xunit", choices=("cm", "um"), default="cm", help="Radius unit")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=140, help="Output DPI (default: 140)")
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> int:
    run_a = RunDir.find(args.run_a)
    run_b = RunDir.find(args.run_b)
    fields = _parse_csv(args.fields)
    render_compare(
        run_a,
        run_b,
        out_path=args.out,
        dpi=args.dpi,
        xunit=args.xunit,
        at=args.at,
        fields=fields,
    )
    return 0


def render_compare(
    run_a: RunDir,
    run_b: RunDir,
    out_path: str | Path | None,
    dpi: int,
    xunit: str,
    at: str | float | None = None,
    fields: Sequence[str] = DEFAULT_FIELDS,
) -> Path:
    if dpi <= 0:
        raise ValueError("--dpi must be > 0")
    if not fields:
        raise ValueError("at least one field is required")
    for field in fields:
        if field not in FIELD_META:
            raise ValueError(f"unknown field: {field}")

    target = at if at is not None else min(_last_time(run_a), _last_time(run_b))
    snapshot_a = Snapshot.load(pick_snapshots(run_a, at=target)[0])
    snapshot_b = Snapshot.load(pick_snapshots(run_b, at=target)[0])
    history_a = _load_history(run_a)
    history_b = _load_history(run_b)
    include_history = history_a is not None and history_b is not None
    label_a, label_b = _display_labels(run_a, run_b)

    scale, unit = xunit_scale(xunit)
    ncols = len(fields) + int(include_history)
    plt = ensure_matplotlib()
    fig, axes_obj = plt.subplots(
        2,
        ncols,
        figsize=(4.4 * ncols, 6.2),
        squeeze=False,
        sharex=False,
    )

    for col, field in enumerate(fields):
        top = axes_obj[0][col]
        bottom = axes_obj[1][col]
        _plot_compare_field(
            top,
            bottom,
            field,
            label_a,
            label_b,
            snapshot_a,
            snapshot_b,
            scale,
            unit,
        )

    if include_history:
        _plot_history_column(
            axes_obj[0][ncols - 1],
            axes_obj[1][ncols - 1],
            label_a,
            label_b,
            history_a,
            history_b,
        )

    fig.suptitle(f"{label_a} vs {label_b}")
    output = (
        Path(out_path)
        if out_path is not None
        else run_a.path / "plots" / f"compare_{run_a.case}_vs_{run_b.case}.png"
    )
    saved = save_figure(fig, output, dpi)
    plt.close(fig)
    return saved


def _display_labels(run_a: RunDir, run_b: RunDir) -> tuple[str, str]:
    if run_a.case != run_b.case:
        return run_a.case, run_b.case
    return run_a.path.name, run_b.path.name


def _plot_compare_field(
    top,
    bottom,
    field: str,
    label_a: str,
    label_b: str,
    snapshot_a: Snapshot,
    snapshot_b: Snapshot,
    scale: float,
    unit: str,
) -> None:
    x_a, y_a = _field_xy(snapshot_a, field)
    x_b, y_b = _field_xy(snapshot_b, field)
    top.set_title(
        f"{FIELD_META[field][0]}\n"
        f"{label_a} t={format_time(snapshot_a.t)}; "
        f"{label_b} t={format_time(snapshot_b.t)}"
    )
    top.set_ylabel(axis_label(field))
    top.grid(True, alpha=0.25)
    bottom.set_xlabel(f"r [{unit}]")
    bottom.set_ylabel("(B-A)/max(|A|, tiny)")
    bottom.grid(True, alpha=0.25)

    if x_a is None or y_a is None or x_b is None or y_b is None:
        _mark_na(top)
        _mark_na(bottom)
        return

    x_a = np.asarray(x_a, dtype=np.float64)
    x_b = np.asarray(x_b, dtype=np.float64)
    y_a = np.asarray(y_a, dtype=np.float64)
    y_b = np.asarray(y_b, dtype=np.float64)
    top.plot(x_a * scale, y_a, label=label_a, linewidth=1.3)
    top.plot(x_b * scale, y_b, label=label_b, linewidth=1.3, linestyle="--")
    if FIELD_META[field][2]:
        top.set_yscale("log")
    top.legend(fontsize=8)

    y_b_interp = np.interp(x_a, x_b, y_b)
    rel = (y_b_interp - y_a) / np.maximum(np.abs(y_a), TINY)
    bottom.plot(x_a * scale, rel, color="black", linewidth=1.0)


def _plot_history_column(
    top,
    bottom,
    label_a: str,
    label_b: str,
    history_a: History,
    history_b: History,
) -> None:
    time_scale, time_label = time_unit(
        max(
            float(np.max(np.abs(history_a.t))) if len(history_a.t) else 0.0,
            float(np.max(np.abs(history_b.t))) if len(history_b.t) else 0.0,
        )
    )
    top.set_title("history")
    top.set_xlabel(f"t [{time_label}]")
    top.set_ylabel("|conservation_error|")
    top.set_yscale("log")
    top.grid(True, alpha=0.25)
    plotted = False
    cons_a = history_a.get("energy/conservation_error")
    cons_b = history_b.get("energy/conservation_error")
    if cons_a is not None and cons_b is not None:
        for label, history, arr, linestyle in (
            (label_a, history_a, cons_a, "-"),
            (label_b, history_b, cons_b, "--"),
        ):
            top.plot(
                history.t * time_scale,
                np.ma.masked_less_equal(np.abs(np.asarray(arr, dtype=np.float64)), 0.0),
                label=label,
                linestyle=linestyle,
                linewidth=1.1,
            )
            plotted = True
    if plotted:
        top.legend(fontsize=8)
    else:
        _mark_na(top)

    bottom.set_xlabel(f"t [{time_label}]")
    bottom.set_ylabel("absorbed_total [erg]")
    bottom.grid(True, alpha=0.25)
    plotted = False
    absorbed_a = history_a.get("laser/absorbed_total")
    absorbed_b = history_b.get("laser/absorbed_total")
    if absorbed_a is not None and absorbed_b is not None:
        for label, history, arr, linestyle in (
            (label_a, history_a, absorbed_a, "-"),
            (label_b, history_b, absorbed_b, "--"),
        ):
            bottom.plot(
                history.t * time_scale,
                np.asarray(arr, dtype=np.float64),
                label=label,
                linestyle=linestyle,
                linewidth=1.1,
            )
            plotted = True
    if plotted:
        bottom.legend(fontsize=8)
    else:
        _mark_na(bottom)


def _field_xy(snapshot: Snapshot, field: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    if field == "u":
        return snapshot.x_r, snapshot.v_r
    if field == "Tr":
        if snapshot.rad_energy_density is None:
            return None, None
        return snapshot.r_centers, derived.radiation_temperature(snapshot.rad_energy_density)
    if field == "P":
        pe = snapshot.field("Pe")
        pi = snapshot.field("Pi")
        if pe is None or pi is None:
            return None, None
        return snapshot.r_centers, derived.total_pressure(pe, pi)
    if field == "laser_dep":
        return snapshot.r_centers, snapshot.laser_deposited_power
    return snapshot.r_centers, snapshot.field(field)


def _load_history(run_dir: RunDir) -> History | None:
    if run_dir.history_path is None:
        return None
    return History.load(run_dir.history_path)


def _last_time(run_dir: RunDir) -> float:
    return float(run_dir.snapshot_times()[-1][1])


def _parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _mark_na(ax) -> None:
    ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center", va="center")
