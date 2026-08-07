"""Spacetime plot command."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .. import derived
from ..io import RunDir, Snapshot
from ..style import FIELD_META, axis_label, ensure_matplotlib, save_figure, time_unit, xunit_scale


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("spacetime", help="Plot an r-t cell-field map.")
    parser.add_argument("run", type=Path, help="TENRYU output directory or results/ directory")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=140, help="Output DPI (default: 140)")
    parser.add_argument("--xunit", choices=("cm", "um"), default="cm", help="Radius unit")
    parser.add_argument("--field", default="rho", help="Cell field to plot")
    parser.add_argument("--log", action="store_true", default=False, help="Use LogNorm on positive data")
    parser.add_argument("--mesh", type=int, default=None, help="Overlay every Nth node trajectory")
    parser.add_argument("--shock", action="store_true", default=False, help="Overlay Qvisc shock-radius estimates")
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> int:
    run_dir = RunDir.find(args.run)
    render_spacetime(
        run_dir,
        out_path=args.out,
        dpi=args.dpi,
        xunit=args.xunit,
        field=args.field,
        log=args.log,
        mesh_every=args.mesh,
        shock=args.shock,
    )
    return 0


def render_spacetime(
    run_dir: RunDir,
    out_path: str | Path | None,
    dpi: int,
    xunit: str,
    field: str = "rho",
    log: bool = False,
    mesh_every: int | None = None,
    shock: bool = False,
) -> Path:
    if dpi <= 0:
        raise ValueError("--dpi must be > 0")
    if field not in FIELD_META:
        raise ValueError(f"unknown field: {field}")
    if field == "u":
        raise ValueError("spacetime requires a cell field; 'u' is nodal")
    if mesh_every is not None and mesh_every <= 0:
        raise ValueError("--mesh must be > 0")

    snapshots = [Snapshot.load(path) for path in run_dir.snapshot_paths]
    X_samples, C = _build_matrices(snapshots, field)
    t_s = np.asarray([snapshot.t for snapshot in snapshots], dtype=np.float64)
    X_edges = _extend_x_edges(X_samples)
    t_scale, t_unit = time_unit(float(np.max(np.abs(t_s))) if t_s.size else 0.0)
    t_edges = _time_edges(t_s, snapshots) * t_scale
    Y_edges = np.broadcast_to(t_edges[:, None], X_edges.shape)
    scale, unit = xunit_scale(xunit)

    plt = ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    norm = None
    C_plot = C
    if log:
        from matplotlib.colors import LogNorm

        C_masked = np.ma.masked_less_equal(C, 0.0)
        if C_masked.count() == 0:
            raise ValueError(f"field {field} has no positive values for --log")
        norm = LogNorm(vmin=float(C_masked.min()), vmax=float(C_masked.max()))
        C_plot = C_masked

    mesh = ax.pcolormesh(X_edges * scale, Y_edges, C_plot, shading="flat", norm=norm)
    fig.colorbar(mesh, ax=ax, label=axis_label(field))
    if mesh_every is not None:
        for node in range(0, X_samples.shape[1], mesh_every):
            ax.plot(X_samples[:, node] * scale, t_s * t_scale, color="0.45", linewidth=0.35, alpha=0.7)
    if shock:
        shock_r: list[float] = []
        shock_t: list[float] = []
        for snapshot in snapshots:
            radius = derived.shock_radius(snapshot)
            if radius is None:
                continue
            shock_r.append(radius * scale)
            shock_t.append(snapshot.t * t_scale)
        if shock_r:
            ax.plot(shock_r, shock_t, "k.", markersize=2.5)

    ax.set_xlabel(f"r [{unit}]")
    ax.set_ylabel(f"t [{t_unit}]")
    ax.set_title(f"{run_dir.case}: {FIELD_META[field][0]} spacetime")
    output = Path(out_path) if out_path is not None else run_dir.path / "plots" / f"spacetime_{field}.png"
    saved = save_figure(fig, output, dpi)
    plt.close(fig)
    return saved


def _build_matrices(snapshots: list[Snapshot], field: str) -> tuple[np.ndarray, np.ndarray]:
    if not snapshots:
        raise ValueError("no snapshots available")
    n_nodes = snapshots[0].x_r.size
    n_cells = snapshots[0].n_cells
    X_rows: list[np.ndarray] = []
    C_rows: list[np.ndarray] = []
    for snapshot in snapshots:
        if snapshot.x_r.size != n_nodes or snapshot.n_cells != n_cells:
            raise ValueError("spacetime requires a fixed cell/node count across snapshots")
        values = _cell_field(snapshot, field)
        if values is None:
            raise ValueError(f"field {field} is not available in {snapshot.path.name}")
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (n_cells,):
            raise ValueError(f"field {field} in {snapshot.path.name} has shape {values.shape}, expected {(n_cells,)}")
        X_rows.append(snapshot.x_r)
        C_rows.append(values)
    return np.vstack(X_rows), np.vstack(C_rows)


def _cell_field(snapshot: Snapshot, field: str) -> np.ndarray | None:
    if field == "Tr":
        if snapshot.rad_energy_density is None:
            return None
        return derived.radiation_temperature(snapshot.rad_energy_density)
    if field == "P":
        pe = snapshot.field("Pe")
        pi = snapshot.field("Pi")
        if pe is None or pi is None:
            return None
        return derived.total_pressure(pe, pi)
    if field == "laser_dep":
        return snapshot.laser_deposited_power
    return snapshot.field(field)


def _time_edges(t_s: np.ndarray, snapshots: list[Snapshot]) -> np.ndarray:
    if t_s.size == 1:
        width = snapshots[0].dt if snapshots[0].dt > 0.0 else 1.0e-12
        return np.array([t_s[0] - 0.5 * width, t_s[0] + 0.5 * width])
    edges = np.empty(t_s.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (t_s[:-1] + t_s[1:])
    edges[0] = t_s[0] - 0.5 * (t_s[1] - t_s[0])
    edges[-1] = t_s[-1] + 0.5 * (t_s[-1] - t_s[-2])
    return edges


def _extend_x_edges(X_samples: np.ndarray) -> np.ndarray:
    if X_samples.shape[0] == 1:
        return np.vstack([X_samples[0], X_samples[0]])
    edges = np.empty((X_samples.shape[0] + 1, X_samples.shape[1]), dtype=np.float64)
    edges[1:-1] = 0.5 * (X_samples[:-1] + X_samples[1:])
    edges[0] = X_samples[0] - 0.5 * (X_samples[1] - X_samples[0])
    edges[-1] = X_samples[-1] + 0.5 * (X_samples[-1] - X_samples[-2])
    return edges
