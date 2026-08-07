"""Summary plot bundle command."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .. import derived
from ..io import History, RunDir, Snapshot
from . import history as history_cmd
from . import profile as profile_cmd
from . import spacetime as spacetime_cmd


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("summary", help="Generate the standard P0 plot bundle.")
    parser.add_argument("run", type=Path, help="TENRYU output directory or results/ directory")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Output directory")
    parser.add_argument("--dpi", type=int, default=140, help="Output DPI (default: 140)")
    parser.add_argument("--xunit", choices=("cm", "um"), default="cm", help="Radius unit")
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> int:
    run_dir = RunDir.find(args.run)
    out_dir = Path(args.out) if args.out is not None else run_dir.path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    profile_cmd.render_profile(
        [run_dir],
        out_path=out_dir / "profile_final.png",
        dpi=args.dpi,
        xunit=args.xunit,
        last=True,
        fields=profile_cmd.DEFAULT_FIELDS,
    )
    spacetime_cmd.render_spacetime(
        run_dir,
        out_path=out_dir / "spacetime_rho.png",
        dpi=args.dpi,
        xunit=args.xunit,
        field="rho",
    )
    spacetime_cmd.render_spacetime(
        run_dir,
        out_path=out_dir / "spacetime_Te.png",
        dpi=args.dpi,
        xunit=args.xunit,
        field="Te",
    )

    history = _load_history_if_present(run_dir)
    if history is not None and history_cmd.panel_available(history, "energy"):
        history_cmd.render_history(
            run_dir,
            out_path=out_dir / "history_energy.png",
            dpi=args.dpi,
            xunit=args.xunit,
            panels=["energy"],
            history=history,
        )
    if history is not None and history_cmd.panel_available(history, "laser"):
        history_cmd.render_history(
            run_dir,
            out_path=out_dir / "history_laser.png",
            dpi=args.dpi,
            xunit=args.xunit,
            panels=["laser"],
            history=history,
        )
    if history is not None and history_cmd.panel_available(history, "implosion"):
        history_cmd.render_history(
            run_dir,
            out_path=out_dir / "history_implosion.png",
            dpi=args.dpi,
            xunit=args.xunit,
            panels=["implosion"],
            history=history,
        )

    final_snapshot = Snapshot.load(run_dir.snapshot_paths[-1])
    text = build_summary_text(run_dir, final_snapshot, history)
    summary_path = out_dir / "summary.txt"
    summary_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def build_summary_text(run_dir: RunDir, final_snapshot: Snapshot, history: History | None) -> str:
    run_info = run_dir.run_info or {}
    final_t = _maybe_float(run_info.get("t"))
    final_step = run_info.get("step")
    if final_t is None:
        final_t = final_snapshot.t
    if final_step is None:
        final_step = final_snapshot.step

    lines = [
        f"case: {run_dir.case}",
        f"geometry: {final_snapshot.geometry} ({final_snapshot.geometry_source})",
        f"n_cells: {final_snapshot.n_cells}",
        f"n_groups: {final_snapshot.n_groups}",
        f"n_materials: {final_snapshot.n_materials}",
        f"termination_reason: {_fmt_value(run_info.get('termination_reason'))}",
        f"final_t_s: {_fmt_float(final_t)}",
        f"final_step: {_fmt_value(final_step)}",
    ]

    conservation = history.get("energy/conservation_error") if history is not None else None
    if conservation is not None:
        values = np.asarray(conservation, dtype=np.float64)
        lines.append(f"energy_conservation_error_final: {_fmt_float(_last_value(values))}")
        lines.append(f"energy_conservation_error_max_abs: {_fmt_float(_max_abs(values))}")
    else:
        lines.append("energy_conservation_error_final: n/a")
        lines.append("energy_conservation_error_max_abs: n/a")

    laser_absorbed = history.get("laser/absorbed_total") if history is not None else None
    if laser_absorbed is not None:
        lines.append(f"laser_absorbed_total_erg: {_fmt_float(_last_value(laser_absorbed))}")
    else:
        lines.append("laser_absorbed_total_erg: n/a")

    laser_fraction = history.get("laser/absorption_efficiency_total") if history is not None else None
    if laser_fraction is not None:
        lines.append(f"laser_absorption_fraction: {_fmt_float(_last_value(laser_fraction))}")
    elif final_snapshot.laser_absorption_fraction is not None:
        lines.append(f"laser_absorption_fraction: {_fmt_float(final_snapshot.laser_absorption_fraction)}")
    else:
        lines.append("laser_absorption_fraction: n/a")

    bang = derived.bang_time(history) if history is not None else None
    lines.append(f"bang_time_s: {_fmt_float(bang) if bang is not None else 'n/a'}")
    rho_peak = history.get("implosion/rho_peak") if history is not None else None
    if rho_peak is not None:
        lines.append(f"rho_peak_max_g_cm3: {_fmt_float(_max_value(rho_peak))}")
    else:
        lines.append("rho_peak_max_g_cm3: n/a")

    shock = derived.shock_radius(final_snapshot)
    lines.append(f"final_shock_radius_cm: {_fmt_float(shock) if shock is not None else 'n/a'}")
    return "\n".join(lines)


def _load_history_if_present(run_dir: RunDir) -> History | None:
    if run_dir.history_path is None:
        return None
    return History.load(run_dir.history_path)


def _maybe_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _fmt_value(value) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _fmt_float(value: float) -> str:
    return f"{float(value):.8e}"


def _last_value(arr) -> float:
    values = np.asarray(arr, dtype=np.float64)
    if values.ndim > 1:
        values = values[:, 0]
    return float(values[-1])


def _max_abs(arr) -> float:
    values = np.asarray(arr, dtype=np.float64)
    return float(np.nanmax(np.abs(values)))


def _max_value(arr) -> float:
    values = np.asarray(arr, dtype=np.float64)
    return float(np.nanmax(values))
