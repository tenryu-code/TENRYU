"""Profile plot command."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .. import derived
from ..io import RunDir, Snapshot, pick_snapshots
from ..style import FIELD_META, axis_label, ensure_matplotlib, format_time, save_figure, xunit_scale


DEFAULT_FIELDS = ("rho", "Te", "Ti", "Tr", "u", "P")


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("profile", help="Plot radial profiles from one snapshot.")
    parser.add_argument("run", type=Path, nargs="+", help="TENRYU output directory or results/ directory")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=140, help="Output DPI (default: 140)")
    parser.add_argument("--xunit", choices=("cm", "um"), default="cm", help="Radius unit")
    parser.add_argument("--at", default=None, help="Time selector, e.g. 1.2ns or 3e-9")
    parser.add_argument("--index", type=int, default=None, help="4-digit snapshot file index")
    parser.add_argument("--last", action="store_true", default=False, help="Use the last snapshot")
    parser.add_argument("--fields", default=",".join(DEFAULT_FIELDS), help="Comma-separated fields")
    parser.add_argument("--logy", default="", help="Comma-separated fields forced to log-y")
    parser.add_argument("--window", default=None, help="Displayed x-range r0,r1 in the selected xunit")
    parser.add_argument("--overlay", choices=("noh", "sod", "rmtv"), default=None, help="Overlay exact reference")
    parser.add_argument("--overlay-params", default="", help="Comma-separated exact-reference parameters k=v")
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> int:
    render_profile(
        args.run,
        out_path=args.out,
        dpi=args.dpi,
        xunit=args.xunit,
        at=args.at,
        index=args.index,
        last=args.last or (args.at is None and args.index is None),
        fields=_parse_csv(args.fields),
        logy_fields=set(_parse_csv(args.logy)),
        window=_parse_window(args.window),
        overlay=args.overlay,
        overlay_params=_parse_overlay_params(args.overlay, args.overlay_params),
    )
    return 0


def render_profile(
    runs: Sequence[str | Path | RunDir],
    out_path: str | Path | None,
    dpi: int,
    xunit: str,
    at: str | float | None = None,
    index: int | None = None,
    last: bool = True,
    fields: Sequence[str] = DEFAULT_FIELDS,
    logy_fields: set[str] | None = None,
    window: tuple[float, float] | None = None,
    overlay: str | None = None,
    overlay_params: dict[str, float] | None = None,
) -> Path:
    if dpi <= 0:
        raise ValueError("--dpi must be > 0")
    if not runs:
        raise ValueError("at least one run is required")
    logy_fields = logy_fields or set()
    run_dirs = [_as_run_dir(run_path) for run_path in runs]
    snapshots = [
        Snapshot.load(pick_snapshots(run_dir, at=at, index=index, last=last)[0])
        for run_dir in run_dirs
    ]
    plot_fields, tr_missing = _resolved_fields(fields, snapshots)
    if not plot_fields:
        raise ValueError("no plottable fields were selected")

    plt = ensure_matplotlib()
    n_panels = len(plot_fields)
    ncols = min(3, n_panels)
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes_obj = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.3 * nrows), squeeze=False)
    axes = [axes_obj[row][col] for row in range(nrows) for col in range(ncols)]
    scale, unit = xunit_scale(xunit)
    interfaces = _material_interfaces(snapshots[0], scale)
    overlay_data = _reference_overlay(overlay, overlay_params or {}, snapshots[0], scale, window)
    overlay_fields = set(overlay_data[1].keys()) if overlay_data is not None else set()

    for ax, field in zip(axes, plot_fields):
        _plot_field(ax, field, run_dirs, snapshots, scale)
        if overlay_data is not None and field in overlay_fields:
            r_dense, values = overlay_data
            ax.plot(
                r_dense * scale,
                values[field],
                color="black",
                linestyle="--",
                linewidth=1.0,
                label="exact",
            )
        for x in interfaces:
            ax.axvline(x, color="0.25", linestyle=":", linewidth=0.7, alpha=0.55)
        if field in logy_fields or (field not in logy_fields and FIELD_META[field][2]):
            ax.set_yscale("log")
        if window is not None:
            ax.set_xlim(window)
        ax.set_xlabel(f"r [{unit}]")
        ax.set_ylabel(axis_label(field))
        ax.set_title(FIELD_META[field][0])
        ax.grid(True, alpha=0.25)
        if len(run_dirs) > 1 or field == "P" or field in overlay_fields:
            ax.legend(fontsize=8)

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    title = _profile_title(run_dirs, snapshots)
    if tr_missing:
        title += " (Tr n/a)"
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output = Path(out_path) if out_path is not None else _default_output(run_dirs[0], snapshots[0])
    saved = save_figure(fig, output, dpi)
    plt.close(fig)
    return saved


def _as_run_dir(value: str | Path | RunDir) -> RunDir:
    if isinstance(value, RunDir):
        return value
    return RunDir.find(value)


def _parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_window(value: str | None) -> tuple[float, float] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError("--window must be r0,r1")
    lo = float(parts[0])
    hi = float(parts[1])
    if hi <= lo:
        raise ValueError("--window upper bound must be greater than lower bound")
    return lo, hi


def _parse_overlay_params(overlay: str | None, value: str | None) -> dict[str, float]:
    if overlay is None:
        if value:
            raise ValueError("--overlay-params requires --overlay")
        return {}
    allowed = {
        "noh": {"rho0", "v0", "gamma"},
        "sod": {"rho_l", "p_l", "rho_r", "p_r", "gamma", "x0"},
        "rmtv": set(),
    }[overlay]
    params: dict[str, float] = {}
    if not value:
        return params
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError("--overlay-params entries must be k=v")
        key, raw = [piece.strip() for piece in text.split("=", 1)]
        if key not in allowed:
            raise ValueError(f"unknown {overlay} overlay parameter: {key}")
        params[key] = float(raw)
    return params


def _resolved_fields(fields: Sequence[str], snapshots: Sequence[Snapshot]) -> tuple[list[str], bool]:
    resolved: list[str] = []
    tr_missing = False
    for field in fields:
        if field not in FIELD_META:
            raise ValueError(f"unknown field: {field}")
        if field == "Tr" and not any(snapshot.rad_energy_density is not None for snapshot in snapshots):
            tr_missing = True
            continue
        if field == "Tr" and any(snapshot.rad_energy_density is None for snapshot in snapshots):
            tr_missing = True
        resolved.append(field)
    return resolved, tr_missing


def _plot_field(
    ax,
    field: str,
    run_dirs: Sequence[RunDir],
    snapshots: Sequence[Snapshot],
    scale: float,
) -> None:
    plotted = False
    for run_dir, snapshot in zip(run_dirs, snapshots):
        label = run_dir.case
        if field == "P":
            pe = snapshot.field("Pe")
            pi = snapshot.field("Pi")
            if pe is not None and pi is not None:
                ax.plot(snapshot.r_centers * scale, derived.total_pressure(pe, pi), label=label, linewidth=1.3)
                plotted = True
            qvisc = snapshot.field("Qvisc")
            if qvisc is not None:
                ax.plot(
                    snapshot.r_centers * scale,
                    qvisc,
                    linestyle="--",
                    linewidth=1.0,
                    label=f"{label} Qvisc",
                )
                plotted = True
            continue

        x, y = _field_xy(snapshot, field)
        if x is None or y is None:
            continue
        ax.plot(x * scale, y, label=label, linewidth=1.3)
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center", va="center")


def _field_xy(snapshot: Snapshot, field: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    if field == "u":
        return snapshot.x_r, snapshot.v_r
    if field == "Tr":
        if snapshot.rad_energy_density is None:
            return None, None
        return snapshot.r_centers, derived.radiation_temperature(snapshot.rad_energy_density)
    if field == "laser_dep":
        return snapshot.r_centers, snapshot.laser_deposited_power
    return snapshot.r_centers, snapshot.field(field)


def _reference_overlay(
    overlay: str | None,
    params: dict[str, float],
    snapshot: Snapshot,
    scale: float,
    window: tuple[float, float] | None,
) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    if overlay is None:
        return None
    if window is None:
        lo = float(np.min(snapshot.x_r))
        hi = float(np.max(snapshot.x_r))
    else:
        lo = window[0] / scale
        hi = window[1] / scale
    r_dense = np.linspace(lo, hi, 2048)
    if overlay == "noh":
        from ..refs import noh

        ref = noh.profile(r_dense, snapshot.t, snapshot.geometry, **params)
    elif overlay == "sod":
        from ..refs.sod import SodExact

        x0 = params.get("x0", 0.0)
        state_params = {key: value for key, value in params.items() if key != "x0"}
        ref = SodExact(**state_params).profile(r_dense, snapshot.t, x0=x0)
    elif overlay == "rmtv":
        from ..refs import rmtv

        ref = rmtv.profile(r_dense, snapshot.t)
        # RMtV overlay deliberately exposes only rho and Te; the C++ gate certifies Theta and G only.
        return r_dense, {"rho": ref["rho"], "Te": ref["Te"]}
    else:
        raise ValueError(f"unknown overlay: {overlay}")
    return r_dense, {"rho": ref["rho"], "u": ref["u"], "P": ref["p"]}


def _material_interfaces(snapshot: Snapshot, scale: float) -> list[float]:
    material = snapshot.material_id
    if material.size < 2:
        return []
    indices = np.nonzero(material[1:] != material[:-1])[0] + 1
    return [float(snapshot.x_r[index] * scale) for index in indices]


def _profile_title(run_dirs: Sequence[RunDir], snapshots: Sequence[Snapshot]) -> str:
    parts = [
        f"{run_dir.case}: t={format_time(snapshot.t)}, step={snapshot.step}"
        for run_dir, snapshot in zip(run_dirs, snapshots)
    ]
    return "; ".join(parts)


def _default_output(run_dir: RunDir, snapshot: Snapshot) -> Path:
    return run_dir.path / "plots" / f"profile_{_time_stamp(snapshot.t)}.png"


def _time_stamp(t_s: float) -> str:
    return f"t{format_time(t_s).replace(' ', '')}"
