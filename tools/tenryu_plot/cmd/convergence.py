"""Convergence plot command."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .. import derived
from ..io import RunDir, Snapshot, pick_snapshots
from ..style import FIELD_META, axis_label, ensure_matplotlib, save_figure


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("convergence", help="Plot profile error convergence.")
    parser.add_argument("runs", type=Path, nargs="+", help="Sample TENRYU output directories")
    ref = parser.add_mutually_exclusive_group(required=True)
    ref.add_argument("--ref", choices=("noh", "sod", "rmtv"), default=None, help="Exact reference")
    ref.add_argument("--ref-run", type=Path, default=None, help="Reference TENRYU output directory")
    parser.add_argument("--overlay-params", default="", help="Comma-separated exact-reference parameters k=v")
    parser.add_argument("--field", default="rho", help="Field to compare")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--at", default=None, help="Time selector, e.g. 1.2ns or 3e-9")
    selector.add_argument("--last", action="store_true", default=False, help="Use the last snapshot")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=140, help="Output DPI (default: 140)")
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> int:
    if len(args.runs) < 2:
        raise ValueError("convergence requires at least two sample runs")
    if args.ref_run is not None and args.overlay_params:
        raise ValueError("--overlay-params requires --ref")
    sample_runs = [RunDir.find(path) for path in args.runs]
    ref_run = RunDir.find(args.ref_run) if args.ref_run is not None else None
    params = _parse_overlay_params(args.ref, args.overlay_params) if args.ref is not None else {}
    render_convergence(
        sample_runs,
        ref=args.ref,
        ref_run=ref_run,
        out_path=args.out,
        dpi=args.dpi,
        field=args.field,
        at=args.at,
        overlay_params=params,
    )
    return 0


def render_convergence(
    runs: Sequence[RunDir],
    ref: str | None,
    ref_run: RunDir | None,
    out_path: str | Path | None,
    dpi: int,
    field: str = "rho",
    at: str | float | None = None,
    overlay_params: dict[str, float] | None = None,
) -> Path:
    if dpi <= 0:
        raise ValueError("--dpi must be > 0")
    if field not in FIELD_META:
        raise ValueError(f"unknown field: {field}")
    exact_fields = {"rho", "Te"} if ref == "rmtv" else {"rho", "u", "P"}
    if ref is not None and field not in exact_fields:
        raise ValueError(f"exact {ref} reference does not define field: {field}")

    snapshots = [_pick_snapshot(run_dir, at) for run_dir in runs]
    ref_snapshot = _pick_snapshot(ref_run, at) if ref_run is not None else None
    if ref_snapshot is not None:
        max_sample_cells = max(snapshot.n_cells for snapshot in snapshots)
        if ref_snapshot.n_cells <= max_sample_cells:
            raise ValueError("--ref-run n_cells must be strictly greater than every sample run")

    rows: list[tuple[int, float, float]] = []
    for snapshot in snapshots:
        x, values, dr = _sample_values(snapshot, field)
        if ref_snapshot is not None:
            x_ref, values_ref, _ = _sample_values(ref_snapshot, field)
            ref_values = np.interp(x, x_ref, values_ref)
        elif ref is not None:
            ref_values = _exact_reference(ref, overlay_params or {}, snapshot, field, x)
        else:
            raise ValueError("reference is required")
        err = np.asarray(values, dtype=np.float64) - np.asarray(ref_values, dtype=np.float64)
        dr = np.asarray(dr, dtype=np.float64)
        den = np.sum(dr)
        l1 = float(np.sum(np.abs(err) * dr) / den)
        l2 = float(math.sqrt(np.sum(err * err * dr) / den))
        rows.append((snapshot.n_cells, l1, l2))

    print("n_cells L1 L2")
    for n_cells, l1, l2 in rows:
        print(f"{n_cells} {l1:.8e} {l2:.8e}")

    n = np.asarray([row[0] for row in rows], dtype=np.float64)
    l1_values = np.asarray([row[1] for row in rows], dtype=np.float64)
    l2_values = np.asarray([row[2] for row in rows], dtype=np.float64)
    # order p = -slope of log(err) vs log(n)
    p_l1 = -_fit_slope(n, l1_values)
    p_l2 = -_fit_slope(n, l2_values)

    plt = ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.loglog(n, l1_values, "o-", label=f"L1 (p={p_l1:.2f})")
    ax.loglog(n, l2_values, "s-", label=f"L2 (p={p_l2:.2f})")
    ax.set_xlabel("n_cells")
    ax.set_ylabel(f"error in {axis_label(field)}")
    ax.set_title(f"{field} convergence")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    output = Path(out_path) if out_path is not None else runs[0].path / "plots" / f"convergence_{field}.png"
    saved = save_figure(fig, output, dpi)
    plt.close(fig)
    return saved


def _pick_snapshot(run_dir: RunDir | None, at: str | float | None) -> Snapshot:
    if run_dir is None:
        raise ValueError("reference run is required")
    return Snapshot.load(pick_snapshots(run_dir, at=at, last=at is None)[0])


def _sample_values(snapshot: Snapshot, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if field == "u":
        return snapshot.x_r, snapshot.v_r, np.gradient(snapshot.x_r)
    if field == "Tr":
        if snapshot.rad_energy_density is None:
            raise ValueError(f"field {field} is not available in {snapshot.path.name}")
        return snapshot.r_centers, derived.radiation_temperature(snapshot.rad_energy_density), snapshot.dr
    if field == "P":
        pe = snapshot.field("Pe")
        pi = snapshot.field("Pi")
        if pe is None or pi is None:
            raise ValueError(f"field {field} is not available in {snapshot.path.name}")
        return snapshot.r_centers, derived.total_pressure(pe, pi), snapshot.dr
    if field == "laser_dep":
        if snapshot.laser_deposited_power is None:
            raise ValueError(f"field {field} is not available in {snapshot.path.name}")
        return snapshot.r_centers, snapshot.laser_deposited_power, snapshot.dr
    values = snapshot.field(field)
    if values is None:
        raise ValueError(f"field {field} is not available in {snapshot.path.name}")
    return snapshot.r_centers, values, snapshot.dr


def _exact_reference(
    ref: str,
    params: dict[str, float],
    snapshot: Snapshot,
    field: str,
    x: np.ndarray,
) -> np.ndarray:
    key = "p" if field == "P" else field
    if ref == "noh":
        from ..refs import noh

        return noh.profile(x, snapshot.t, snapshot.geometry, **params)[key]
    if ref == "sod":
        from ..refs.sod import SodExact

        x0 = params.get("x0", 0.0)
        state_params = {name: value for name, value in params.items() if name != "x0"}
        return SodExact(**state_params).profile(x, snapshot.t, x0=x0)[key]
    if ref == "rmtv":
        from ..refs import rmtv

        return rmtv.profile(x, snapshot.t)[field]
    raise ValueError(f"unknown reference: {ref}")


def _parse_overlay_params(ref: str | None, value: str | None) -> dict[str, float]:
    if ref is None:
        if value:
            raise ValueError("--overlay-params requires --ref")
        return {}
    allowed = {
        "noh": {"rho0", "v0", "gamma"},
        "sod": {"rho_l", "p_l", "rho_r", "p_r", "gamma", "x0"},
        "rmtv": set(),
    }[ref]
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
            raise ValueError(f"unknown {ref} overlay parameter: {key}")
        params[key] = float(raw)
    return params


def _fit_slope(n_cells: np.ndarray, err: np.ndarray) -> float:
    mask = np.isfinite(err) & (err > 0.0) & np.isfinite(n_cells) & (n_cells > 0.0)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    return float(np.polyfit(np.log(n_cells[mask]), np.log(err[mask]), 1)[0])
