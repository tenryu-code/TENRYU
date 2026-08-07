"""Multigroup radiation spectrum command."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..io import RunDir, Snapshot, pick_snapshots
from ..style import ensure_matplotlib, format_time, save_figure


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("spectrum", help="Plot multigroup radiation spectrum.")
    parser.add_argument("run", type=Path, help="TENRYU output directory or results/ directory")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--at", default=None, help="Time selector, e.g. 1.2ns or 3e-9")
    selector.add_argument("--index", type=int, default=None, help="4-digit snapshot file index")
    selector.add_argument("--last", action="store_true", default=False, help="Use the last snapshot")
    cells = parser.add_mutually_exclusive_group()
    cells.add_argument("--cells", default=None, help="Comma-separated cell indices")
    cells.add_argument("--at-r", default=None, help="Comma-separated radii in cm")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=140, help="Output DPI (default: 140)")
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> int:
    run_dir = RunDir.find(args.run)
    snapshot = Snapshot.load(
        pick_snapshots(
            run_dir,
            at=args.at,
            index=args.index,
            last=args.last or (args.at is None and args.index is None),
        )[0]
    )
    render_spectrum(
        run_dir,
        snapshot,
        out_path=args.out,
        dpi=args.dpi,
        cells_arg=args.cells,
        at_r_arg=args.at_r,
    )
    return 0


def render_spectrum(
    run_dir: RunDir,
    snapshot: Snapshot,
    out_path: str | Path | None,
    dpi: int,
    cells_arg: str | None = None,
    at_r_arg: str | None = None,
) -> Path:
    if dpi <= 0:
        raise ValueError("--dpi must be > 0")
    if snapshot.rad_energy_density is None:
        raise ValueError("run has no radiation/energy_density")
    if snapshot.n_groups == 1:
        raise ValueError("run has n_groups==1; grey -- nothing to plot")
    bounds = snapshot.group_bounds_eV
    if bounds is None:
        raise ValueError("snapshot has no metadata/group_bounds_eV")
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.shape != (snapshot.n_groups + 1,):
        raise ValueError("metadata/group_bounds_eV must have length n_groups + 1")
    energy_density = np.asarray(snapshot.rad_energy_density, dtype=np.float64)
    if energy_density.shape != (snapshot.n_cells, snapshot.n_groups):
        raise ValueError(
            f"radiation/energy_density has shape {energy_density.shape}, "
            f"expected {(snapshot.n_cells, snapshot.n_groups)}"
        )

    cells = _select_cells(snapshot, cells_arg, at_r_arg)
    centers = np.sqrt(bounds[:-1] * bounds[1:])

    plt = ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for cell in cells:
        values = energy_density[cell, :]
        ax.stairs(values, edges=bounds, label=f"r={snapshot.r_centers[cell]:.6g} cm (cell {cell})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    if np.any(centers > 0.0):
        ax.set_xlim(float(np.min(centers[centers > 0.0])), float(np.max(bounds)))
    ax.set_xlabel("group energy [eV]")
    ax.set_ylabel("energy_density [erg/cm^3]")
    ax.set_title(f"{run_dir.case}: spectrum t={format_time(snapshot.t)}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    output = (
        Path(out_path)
        if out_path is not None
        else run_dir.path / "plots" / f"spectrum_{_time_stamp(snapshot.t)}.png"
    )
    saved = save_figure(fig, output, dpi)
    plt.close(fig)
    return saved


def _select_cells(snapshot: Snapshot, cells_arg: str | None, at_r_arg: str | None) -> list[int]:
    if cells_arg is not None:
        cells = [int(part.strip()) for part in cells_arg.split(",") if part.strip()]
    elif at_r_arg is not None:
        radii = [float(part.strip()) for part in at_r_arg.split(",") if part.strip()]
        cells = [int(np.argmin(np.abs(snapshot.r_centers - radius))) for radius in radii]
    else:
        te = snapshot.field("Te")
        if te is None:
            raise ValueError("default spectrum cell requires hydro/Te")
        cells = [int(np.argmax(np.asarray(te, dtype=np.float64)))]
    if not cells:
        raise ValueError("at least one cell is required")
    for cell in cells:
        if cell < 0 or cell >= snapshot.n_cells:
            raise ValueError(f"cell index out of range: {cell}")
    return list(dict.fromkeys(cells))


def _time_stamp(t_s: float) -> str:
    return f"t{format_time(t_s).replace(' ', '')}"
