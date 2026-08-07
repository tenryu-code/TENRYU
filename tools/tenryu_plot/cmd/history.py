"""History plot command."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from .. import derived
from ..io import History, RunDir
from ..style import ensure_matplotlib, save_figure, time_unit, xunit_scale


PANEL_ORDER = ("energy", "conservation", "laser", "implosion", "dt")
ENERGY_COMPONENTS = (
    "energy/internal_electron",
    "energy/internal_ion",
    "energy/kinetic",
    "energy/radiation_field",
    "energy/laser_deposited",
    "energy/laser_escaped",
    "energy/laser_incident",
    "energy/radiation_escaped",
    "energy/numerical_loss",
    "energy/pdv_boundary",
    "energy/floor_injected",
    "energy/safety_injected",
    "energy/solver_residual",
    "energy/E_total",
    "energy/dE_total",
    "energy/E_denom",
    "energy/marshak_in",
    "energy/E_volume_in",
)


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("history", help="Plot time-history diagnostics.")
    parser.add_argument("run", type=Path, help="TENRYU output directory or results/ directory")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=140, help="Output DPI (default: 140)")
    parser.add_argument("--xunit", choices=("cm", "um"), default="cm", help="Radius unit")
    parser.add_argument("--panels", default=None, help="Comma-separated panel list")
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> int:
    run_dir = RunDir.find(args.run)
    history = load_history(run_dir)
    panels = _parse_panels(args.panels) if args.panels else auto_panels(history)
    render_history(run_dir, out_path=args.out, dpi=args.dpi, xunit=args.xunit, panels=panels, history=history)
    return 0


def load_history(run_dir: RunDir) -> History:
    if run_dir.history_path is None:
        raise FileNotFoundError(f"history file not found in {run_dir.results_dir}")
    return History.load(run_dir.history_path)


def auto_panels(history: History) -> list[str]:
    return [panel for panel in PANEL_ORDER if panel_available(history, panel)]


def panel_available(history: History, panel: str) -> bool:
    if panel == "energy":
        return bool(history.group("energy/"))
    if panel == "conservation":
        return bool(history.group("diagnostics/conservation/v1/"))
    if panel == "laser":
        return bool(history.group("laser/"))
    if panel == "implosion":
        return bool(history.group("implosion/"))
    if panel == "dt":
        return bool(history.group("diagnostics/dt_breakdown_history/"))
    raise ValueError(f"unknown history panel: {panel}")


def render_history(
    run_dir: RunDir,
    out_path: str | Path | None,
    dpi: int,
    xunit: str,
    panels: Sequence[str],
    history: History | None = None,
) -> Path:
    if dpi <= 0:
        raise ValueError("--dpi must be > 0")
    history = history if history is not None else load_history(run_dir)
    panels = list(panels)
    if not panels:
        raise ValueError("no history panels available")
    for panel in panels:
        if panel not in PANEL_ORDER:
            raise ValueError(f"unknown history panel: {panel}")

    plt = ensure_matplotlib()
    fig, axes_obj = plt.subplots(len(panels), 1, figsize=(9.0, 3.2 * len(panels)), squeeze=False)
    axes = [axes_obj[row][0] for row in range(len(panels))]
    time_scale, time_label = time_unit(history.t[-1] if len(history.t) else 0)
    time_values = np.asarray(history.t, dtype=np.float64) * time_scale

    for ax, panel in zip(axes, panels):
        if panel == "energy":
            _plot_energy(ax, history, time_values)
        elif panel == "conservation":
            _plot_conservation(ax, history, time_values)
        elif panel == "laser":
            _plot_laser(ax, history, time_values, xunit)
        elif panel == "implosion":
            _plot_implosion(ax, history, time_values, time_scale, xunit)
        elif panel == "dt":
            _plot_dt(ax, history, time_values)
        ax.set_xlabel(f"t [{time_label}]")
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"{run_dir.case} history")
    output = Path(out_path) if out_path is not None else run_dir.path / "plots" / f"history_{'_'.join(panels)}.png"
    saved = save_figure(fig, output, dpi)
    plt.close(fig)
    return saved


def _parse_panels(value: str) -> list[str]:
    panels = [part.strip() for part in value.split(",") if part.strip()]
    for panel in panels:
        if panel not in PANEL_ORDER:
            raise ValueError(f"unknown history panel: {panel}")
    return panels


def _plot_energy(ax, history: History, time_values: np.ndarray) -> None:
    plotted = False
    for name in ENERGY_COMPONENTS:
        plotted = _plot_series(ax, time_values, history.get(name), _short_name(name)) or plotted
    ax.set_ylabel("Energy [erg]")
    err = history.get("energy/conservation_error")
    if err is not None:
        twin = ax.twinx()
        twin.plot(time_values, _positive_abs(err), color="black", linestyle="--", linewidth=1.0, label="conservation_error")
        twin.set_yscale("log")
        twin.set_ylabel("|conservation_error|")
        twin.legend(fontsize=7, loc="upper right")
        plotted = True
    if plotted:
        ax.legend(loc="center left", bbox_to_anchor=(1.08, 0.5), ncol=2, fontsize="x-small", frameon=False)
    else:
        _mark_na(ax)
    ax.set_title("energy")


def _plot_conservation(ax, history: History, time_values: np.ndarray) -> None:
    prefix = "diagnostics/conservation/v1/"
    plotted = False
    for name, arr in sorted(history.group(prefix).items()):
        plotted = _plot_series(ax, time_values, _positive_abs(arr), name[len(prefix) :]) or plotted
    ax.set_yscale("log")
    ax.set_ylabel("|residual|")
    ax.set_title("conservation")
    if plotted:
        ax.legend(fontsize=7)
    else:
        _mark_na(ax)


def _plot_laser(ax, history: History, time_values: np.ndarray, xunit: str) -> None:
    plotted = False
    for name in (
        "laser/absorbed_power_total",
        "laser/commanded_power_total",
        "laser/unabsorbed_power_total",
    ):
        plotted = _plot_series(ax, time_values, history.get(name), _short_name(name)) or plotted
    ax.set_ylabel("Power [erg/s]")
    critical = history.get("laser/critical_surface_r")
    if critical is not None:
        scale, unit = xunit_scale(xunit)
        twin = ax.twinx()
        _plot_series(twin, time_values, np.asarray(critical) * scale, "critical_surface_r", color="black", linestyle="--")
        twin.set_ylabel(f"r [{unit}]")
        twin.legend(fontsize=7, loc="upper right")
        plotted = True
    ax.set_title("laser")
    if plotted:
        ax.legend(fontsize=7, loc="upper left")
    else:
        _mark_na(ax)


def _plot_implosion(ax, history: History, time_values: np.ndarray, time_scale: float, xunit: str) -> None:
    plotted = False
    rho_peak = history.get("implosion/rho_peak")
    if rho_peak is not None:
        plotted = _plot_series(ax, time_values, _positive_abs(rho_peak), "rho_peak") or plotted
        ax.set_yscale("log")
    ax.set_ylabel("rho_peak [g/cm³]")

    scale, unit = xunit_scale(xunit)
    radius_ax = ax.twinx()
    radius_plotted = False
    for name in ("implosion/shell_radius_mean", "implosion/shell_radius_min"):
        arr = history.get(name)
        if arr is not None:
            radius_plotted = _plot_series(radius_ax, time_values, np.asarray(arr) * scale, _short_name(name)) or radius_plotted
    if radius_plotted:
        radius_ax.set_ylabel(f"r [{unit}]")
        radius_ax.legend(fontsize=7, loc="upper right")

    temp = history.get("implosion/center_temperature")
    if temp is not None:
        temp_ax = ax.twinx()
        temp_ax.spines["right"].set_position(("axes", 1.12))
        _plot_series(temp_ax, time_values, temp, "center_temperature", color="tab:red", linestyle=":")
        temp_ax.set_ylabel("T [eV]")
        temp_ax.legend(fontsize=7, loc="lower right")
        plotted = True

    bang = derived.bang_time(history)
    if bang is not None:
        ax.axvline(bang * time_scale, color="black", linestyle="--", linewidth=0.9, label="bang_time")
        plotted = True
    ax.set_title("implosion")
    if plotted:
        ax.legend(fontsize=7, loc="upper left")
    else:
        _mark_na(ax)


def _plot_dt(ax, history: History, time_values: np.ndarray) -> None:
    prefix = "diagnostics/dt_breakdown_history/"
    plotted = False
    chosen = history.get(prefix + "dt_chosen")
    if chosen is not None:
        plotted = _plot_series(ax, time_values, _positive_abs(chosen), "dt_chosen", linewidth=1.4) or plotted
    for name, arr in sorted(history.group(prefix).items()):
        short = name[len(prefix) :]
        if short in {"dt_chosen", "dt_winner_code"} or not short.startswith("dt_"):
            continue
        plotted = _plot_series(ax, time_values, _positive_abs(arr), short) or plotted
    ax.set_yscale("log")
    ax.set_ylabel("dt [s]")
    ax.set_title("dt")
    if plotted:
        ax.legend(fontsize=7, ncol=2)
    else:
        _mark_na(ax)


def _plot_series(ax, x: np.ndarray, arr, label: str, **kwargs) -> bool:
    if arr is None:
        return False
    values = np.asarray(arr)
    plot_kwargs = {"linewidth": 1.0}
    plot_kwargs.update(kwargs)
    if values.ndim == 1:
        ax.plot(x, values, label=label, **plot_kwargs)
        return True
    if values.ndim == 2 and values.shape[1] == 1:
        ax.plot(x, values[:, 0], label=label, **plot_kwargs)
        return True
    if values.ndim == 2:
        for col in range(values.shape[1]):
            ax.plot(x, values[:, col], label=f"{label}[{col}]", **plot_kwargs)
        return True
    return False


def _positive_abs(arr) -> np.ma.MaskedArray:
    values = np.abs(np.asarray(arr, dtype=np.float64))
    return np.ma.masked_less_equal(values, 0.0)


def _short_name(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def _mark_na(ax) -> None:
    ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center", va="center")
