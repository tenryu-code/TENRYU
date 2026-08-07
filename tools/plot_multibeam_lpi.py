#!/usr/bin/env python3
"""Plot multibeam CBET and hot-electron diagnostics from a TENRYU run."""

import argparse
import json
import pathlib

import numpy as np
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HYDRO_RHO = "/hydro/rho"
HYDRO_VOL = "/hydro/vol"
HYDRO_ZBAR = "/hydro/zbar"
HYDRO_VOLFRAC = "/hydro/volFrac"
HYDRO_CBET_GROSS = "/hydro/cbet_gross_exchange"
HYDRO_CBET_NET = "/hydro/cbet_net_to_inbound"
HYDRO_HOT_E_Q = "/hydro/hot_e_Q"
MESH_X_R = "/mesh/x_r"
PORT_SECTION = "/laser/port_section"

ETA_PATHS = {
    "eta": "/hydro/hot_e_eta_model_eta",
    "eta_eq": "/hydro/hot_e_eta_model_eta_eq",
    "g": "/hydro/hot_e_eta_model_g",
    "tau": "/hydro/hot_e_eta_model_tau",
    "I14": "/hydro/hot_e_eta_model_I14",
    "I14_lower": "/hydro/hot_e_eta_model_I14_lower",
    "I14_upper": "/hydro/hot_e_eta_model_I14_upper",
    "Te": "/hydro/hot_e_eta_model_Te",
    "Ln": "/hydro/hot_e_eta_model_Ln",
}

SKY_PATHS = {
    "mu": f"{PORT_SECTION}/grid_mu",
    "phi": f"{PORT_SECTION}/grid_phi",
    "I_tot": f"{PORT_SECTION}/I_tot_eval",
    "I_cw": f"{PORT_SECTION}/I_cw_eval",
    "n_sigma": f"{PORT_SECTION}/n_sigma_eval",
}

RAY_MAP = f"{PORT_SECTION}/ray_map"
RAY_MAP_SHELL_R = f"{PORT_SECTION}/ray_map_shell_r"
PORT_OUTGOING_POWER = f"{PORT_SECTION}/port_outgoing_power"
PORT_CAPTURE_PCROSS = f"{PORT_SECTION}/port_capture_Pcross"

PI = 3.14159265358979323846
C_LIGHT = 2.99792458e10
ELECTRON_MASS = 9.1094e-28
ELEMENTARY_CHARGE = 4.8032e-10
PROTON_MASS = 1.6726219e-24
ERG_PER_S_TO_W = 1.0e-7


def get(f, path):
    """Return an HDF5 dataset as a NumPy array, or None when it is absent."""
    try:
        return np.asarray(f[path][...])
    except (KeyError, OSError, ValueError):
        return None


def warn(message):
    print(f"WARNING: {message}")


def warn_skipped(figure_name, missing):
    missing_text = ", ".join(sorted(set(missing)))
    warn(f"skipping {figure_name}; missing datasets: {missing_text}")


def written(path, datasets):
    used_text = ", ".join(sorted(set(datasets)))
    print(f"WROTE {path} | datasets: {used_text}")


def snapshot_time(f):
    if "t" in f.attrs:
        value = np.asarray(f.attrs["t"]).reshape(-1)
        if value.size == 1 and np.isfinite(value[0]):
            return float(value[0]), "@t"
    value = get(f, "/time_state/t")
    if value is not None:
        flat = np.asarray(value).reshape(-1)
        if flat.size == 1 and np.isfinite(flat[0]):
            return float(flat[0]), "/time_state/t"
    return None, None


def discover_snapshots(run_dir):
    files = sorted((run_dir / "results").glob("*.h5"))
    snapshots = []
    for path in files:
        try:
            with h5py.File(path, "r") as f:
                time_s, time_source = snapshot_time(f)
        except OSError as exc:
            warn(f"omitting unreadable snapshot {path}: {exc}")
            continue
        if time_s is None:
            warn(f"omitting {path.name}; missing root attribute @t and /time_state/t")
            continue
        snapshots.append(
            {"path": path, "time_s": time_s, "time_source": time_source}
        )
    snapshots.sort(key=lambda item: (item["time_s"], item["path"].name))
    return snapshots


def load_frozen_config(run_dir):
    paths = sorted((run_dir / "config").glob("*_frozen.json"))
    if not paths:
        warn(f"no frozen config found under {run_dir / 'config'}")
        return None, None
    if len(paths) > 1:
        warn(f"multiple frozen configs found; using deterministic first path {paths[0]}")
    path = paths[0]
    try:
        config = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        warn(f"could not read frozen config {path}: {exc}")
        return None, path
    return config, path


def normalized_ports(config):
    if not isinstance(config, dict):
        return None
    laser = config.get("laser")
    if not isinstance(laser, dict):
        return None
    port_configuration = laser.get("port_configuration")
    if not isinstance(port_configuration, dict):
        return None
    port_table = port_configuration.get("ports")
    if not isinstance(port_table, list) or not port_table:
        return None

    axes = []
    weights = []
    ids = []
    for index, port in enumerate(port_table):
        if not isinstance(port, dict):
            return None
        direction = np.asarray(port.get("direction", []), dtype=float)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            return None
        norm = float(np.linalg.norm(direction))
        if not norm > 0.0:
            return None
        weight = port.get("power_weight")
        if not isinstance(weight, (int, float)) or not np.isfinite(weight):
            return None
        axes.append(direction / norm)
        weights.append(float(weight))
        ids.append(port.get("port_id", index))
    return {
        "axes": np.asarray(axes),
        "weights": np.asarray(weights),
        "ids": ids,
    }


def frozen_channels(config):
    if not isinstance(config, dict):
        return []
    laser = config.get("laser")
    if not isinstance(laser, dict):
        return []
    hot_electron = laser.get("hot_electron")
    if not isinstance(hot_electron, dict):
        return []
    sources = hot_electron.get("sources")
    return sources if isinstance(sources, list) else []


def radius_edges(f, n_cells):
    x_r = get(f, MESH_X_R)
    if x_r is not None:
        edges = np.asarray(x_r, dtype=float).reshape(-1)
        if (
            edges.size == n_cells + 1
            and np.all(np.isfinite(edges))
            and np.all(np.diff(edges) > 0.0)
        ):
            return edges, MESH_X_R, []

    geometry = f.attrs.get("geometry")
    if isinstance(geometry, bytes):
        geometry = geometry.decode(errors="replace")
    vol = get(f, HYDRO_VOL)
    if (
        geometry == "1D_SPH"
        and vol is not None
        and np.asarray(vol).reshape(-1).size == n_cells
    ):
        shell_volumes = np.asarray(vol, dtype=float).reshape(-1)
        if np.all(np.isfinite(shell_volumes)) and np.all(shell_volumes >= 0.0):
            cumulative = np.concatenate(([0.0], np.cumsum(shell_volumes)))
            edges = np.cbrt(3.0 * cumulative / (4.0 * PI))
            if np.all(np.diff(edges) > 0.0):
                return edges, HYDRO_VOL, []

    missing = [MESH_X_R]
    if vol is None:
        missing.append(HYDRO_VOL)
    if geometry != "1D_SPH":
        missing.append("@geometry=1D_SPH")
    return None, None, missing


def center_edges(values, lower=None, upper=None):
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        delta = max(abs(float(values[0])) * 0.05, 0.05)
        edges = np.asarray([values[0] - delta, values[0] + delta])
    else:
        edges = np.empty(values.size + 1, dtype=float)
        edges[1:-1] = 0.5 * (values[:-1] + values[1:])
        edges[0] = values[0] - 0.5 * (values[1] - values[0])
        edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    if lower is not None:
        edges[0] = lower
    if upper is not None:
        edges[-1] = upper
    return edges


def selected_snapshots(snapshots, requested_times):
    if not snapshots:
        return []
    if requested_times is None:
        count = min(3, len(snapshots))
        indices = np.rint(np.linspace(0, len(snapshots) - 1, count)).astype(int)
    else:
        times_ns = np.asarray([item["time_s"] * 1.0e9 for item in snapshots])
        indices = np.asarray(
            [int(np.argmin(np.abs(times_ns - target))) for target in requested_times],
            dtype=int,
        )
    selected = []
    seen = set()
    for index in indices:
        index = int(index)
        if index not in seen:
            selected.append(snapshots[index])
            seen.add(index)
    return selected


def time_tag(time_s):
    return f"{time_s * 1.0e9:.3f}"


def load_required(f, paths):
    values = {}
    missing = []
    for key, path in paths.items():
        value = get(f, path)
        if value is None:
            missing.append(path)
        else:
            values[key] = value
    return values, missing


def positive_log_data(values):
    maximum = float(np.nanmax(values)) if values.size else 0.0
    if not np.isfinite(maximum) or maximum <= 0.0:
        floor = 1.0e-30
        maximum = 1.0e-29
    else:
        floor = maximum * 1.0e-8
    plotted = np.where(np.isfinite(values) & (values > floor), values, floor)
    return plotted, floor, maximum


def plot_streaks(snapshots, out_dir, dpi):
    required = {
        "rho": HYDRO_RHO,
        "gross": HYDRO_CBET_GROSS,
        "net": HYDRO_CBET_NET,
        "hot_e_Q": HYDRO_HOT_E_Q,
    }
    rows = []
    missing_all = []
    used = []
    n_cells = None

    for snapshot in snapshots:
        try:
            with h5py.File(snapshot["path"], "r") as f:
                values, missing = load_required(f, required)
                if missing:
                    missing_all.extend(missing)
                    warn(
                        f"streaks.png: omitted {snapshot['path'].name}; "
                        f"missing datasets: {', '.join(missing)}"
                    )
                    continue
                arrays = {
                    key: np.asarray(value, dtype=float).reshape(-1)
                    for key, value in values.items()
                }
                lengths = {array.size for array in arrays.values()}
                if len(lengths) != 1:
                    warn(
                        f"streaks.png: omitted {snapshot['path'].name}; "
                        "required hydro dataset dimensions disagree"
                    )
                    continue
                this_n_cells = lengths.pop()
                edges, radius_source, radius_missing = radius_edges(f, this_n_cells)
                if edges is None:
                    missing_all.extend(radius_missing)
                    warn(
                        f"streaks.png: omitted {snapshot['path'].name}; "
                        f"missing radius data: {', '.join(radius_missing)}"
                    )
                    continue
        except OSError as exc:
            warn(f"streaks.png: omitted unreadable {snapshot['path']}: {exc}")
            continue

        if n_cells is None:
            n_cells = this_n_cells
        if this_n_cells != n_cells:
            warn(
                f"streaks.png: omitted {snapshot['path'].name}; "
                f"cell count {this_n_cells} differs from {n_cells}"
            )
            continue
        rows.append(
            {
                "time_ns": snapshot["time_s"] * 1.0e9,
                "edges_um": edges * 1.0e4,
                **arrays,
            }
        )
        used.extend(required.values())
        used.extend([radius_source, snapshot["time_source"]])

    if not rows:
        warn_skipped("streaks.png", missing_all or list(required.values()))
        return 0

    times_ns = np.asarray([row["time_ns"] for row in rows])
    time_edges = center_edges(times_ns)
    radial_edges = np.asarray([row["edges_um"] for row in rows])
    radial_mesh_edges = np.empty((len(rows) + 1, radial_edges.shape[1]))
    radial_mesh_edges[0] = radial_edges[0]
    radial_mesh_edges[-1] = radial_edges[-1]
    if len(rows) > 1:
        radial_mesh_edges[1:-1] = 0.5 * (
            radial_edges[:-1] + radial_edges[1:]
        )
    time_mesh_edges = np.repeat(
        time_edges[:, np.newaxis], radial_mesh_edges.shape[1], axis=1
    )

    rho = np.asarray([row["rho"] for row in rows])
    positive_rho = rho[np.isfinite(rho) & (rho > 0.0)]
    if positive_rho.size:
        rho_floor = float(np.min(positive_rho))
    else:
        rho_floor = 1.0e-30
    log_rho = np.log10(np.where(rho > rho_floor, rho, rho_floor))

    gross = np.asarray([row["gross"] for row in rows]) * ERG_PER_S_TO_W
    gross_plot, gross_floor, gross_max = positive_log_data(gross)
    net = np.asarray([row["net"] for row in rows]) * ERG_PER_S_TO_W
    net_abs = float(np.nanmax(np.abs(net))) if net.size else 0.0
    if not np.isfinite(net_abs) or net_abs <= 0.0:
        net_abs = 1.0e-30
    hot_e = np.asarray([row["hot_e_Q"] for row in rows]) * ERG_PER_S_TO_W
    hot_e_plot, hot_e_floor, hot_e_max = positive_log_data(hot_e)

    fig, axes = plt.subplots(
        2, 2, figsize=(13.0, 9.0), sharex=True, sharey=True, constrained_layout=True
    )
    panels = [
        (
            log_rho,
            "viridis",
            None,
            "log10 density",
            r"$\log_{10}\rho$ [g cm$^{-3}$]",
        ),
        (
            gross_plot,
            "magma",
            matplotlib.colors.LogNorm(vmin=gross_floor, vmax=gross_max),
            r"CBET gross exchange (LogNorm; floor = $10^{-8}$ max)",
            r"CBET gross [W cm$^{-3}$]",
        ),
        (
            net,
            "RdBu_r",
            matplotlib.colors.Normalize(vmin=-net_abs, vmax=net_abs),
            "CBET net to inbound",
            r"CBET net [W cm$^{-3}$]",
        ),
        (
            hot_e_plot,
            "inferno",
            matplotlib.colors.LogNorm(vmin=hot_e_floor, vmax=hot_e_max),
            r"Hot-electron source (LogNorm; floor = $10^{-8}$ max)",
            r"$Q_{\rm hot\ e}$ [W cm$^{-3}$]",
        ),
    ]
    for ax, (data, cmap, norm, title, colorbar_label) in zip(
        axes.flat, panels
    ):
        mesh = ax.pcolormesh(
            radial_mesh_edges,
            time_mesh_edges,
            data,
            cmap=cmap,
            norm=norm,
            shading="flat",
            rasterized=True,
        )
        fig.colorbar(mesh, ax=ax, label=colorbar_label)
        ax.set_title(title)
        ax.set_xlabel(r"$r$ [$\mu$m]")
        ax.set_ylabel(r"$t$ [ns]")

    path = out_dir / "streaks.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    written(path, used)
    return 1


def plot_eta_channels(snapshots, config, out_dir, dpi):
    required_keys = ("eta", "eta_eq", "g", "tau", "I14", "Te", "Ln")
    required = {key: ETA_PATHS[key] for key in required_keys}
    rows = []
    missing_all = []
    used = []
    n_channels = None

    for snapshot in snapshots:
        try:
            with h5py.File(snapshot["path"], "r") as f:
                values, missing = load_required(f, required)
                if missing:
                    missing_all.extend(missing)
                    warn(
                        f"eta_channels.png: omitted {snapshot['path'].name}; "
                        f"missing datasets: {', '.join(missing)}"
                    )
                    continue
                arrays = {
                    key: np.asarray(value, dtype=float).reshape(-1)
                    for key, value in values.items()
                }
                lengths = {array.size for array in arrays.values()}
                if len(lengths) != 1:
                    warn(
                        f"eta_channels.png: omitted {snapshot['path'].name}; "
                        "required channel dataset dimensions disagree"
                    )
                    continue
                this_n_channels = lengths.pop()
                lower = get(f, ETA_PATHS["I14_lower"])
                upper = get(f, ETA_PATHS["I14_upper"])
        except OSError as exc:
            warn(f"eta_channels.png: omitted unreadable {snapshot['path']}: {exc}")
            continue

        if n_channels is None:
            n_channels = this_n_channels
        if this_n_channels != n_channels:
            warn(
                f"eta_channels.png: omitted {snapshot['path'].name}; "
                f"channel count {this_n_channels} differs from {n_channels}"
            )
            continue
        if lower is None or np.asarray(lower).reshape(-1).size != n_channels:
            lower_array = np.full(n_channels, np.nan)
        else:
            lower_array = np.asarray(lower, dtype=float).reshape(-1)
            used.append(ETA_PATHS["I14_lower"])
        if upper is None or np.asarray(upper).reshape(-1).size != n_channels:
            upper_array = np.full(n_channels, np.nan)
        else:
            upper_array = np.asarray(upper, dtype=float).reshape(-1)
            used.append(ETA_PATHS["I14_upper"])
        rows.append(
            {
                "time_ns": snapshot["time_s"] * 1.0e9,
                "lower": lower_array,
                "upper": upper_array,
                **arrays,
            }
        )
        used.extend(required.values())
        used.append(snapshot["time_source"])

    if not rows or not n_channels:
        warn_skipped("eta_channels.png", missing_all or list(required.values()))
        return 0

    times_ns = np.asarray([row["time_ns"] for row in rows])
    series = {
        key: np.asarray([row[key] for row in rows])
        for key in required_keys
    }
    lower = np.asarray([row["lower"] for row in rows])
    upper = np.asarray([row["upper"] for row in rows])
    channels = frozen_channels(config)

    fig, axes = plt.subplots(
        n_channels,
        4,
        figsize=(15.0, 3.2 * n_channels),
        squeeze=False,
        constrained_layout=True,
    )
    for channel in range(n_channels):
        if channel < len(channels) and isinstance(channels[channel], dict):
            mechanism = channels[channel].get("mechanism", "channel")
            channel_label = f"ch {channel}: {mechanism}"
        else:
            channel_label = f"channel {channel}"

        ax = axes[channel, 0]
        ax.plot(times_ns, series["eta"][:, channel], color="#0072B2", label=r"$\eta$")
        ax.plot(
            times_ns,
            series["eta_eq"][:, channel],
            color="#D55E00",
            linestyle="--",
            label=r"$\eta_{\rm eq}$",
        )
        ax.set_title(f"{channel_label}: conversion")
        ax.set_xlabel(r"$t$ [ns]")
        ax.set_ylabel(r"$\eta$ [1]")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)

        ax = axes[channel, 1]
        ax.plot(times_ns, series["g"][:, channel], color="#009E73")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title("Threshold ratio")
        ax.set_xlabel(r"$t$ [ns]")
        ax.set_ylabel(r"$g$ [1]")
        ax.grid(True, alpha=0.25)

        ax = axes[channel, 2]
        ax.plot(times_ns, series["I14"][:, channel], color="#CC79A7", label=r"$I_{14}$")
        finite_band = np.isfinite(lower[:, channel]) & np.isfinite(upper[:, channel])
        if np.any(finite_band):
            ax.fill_between(
                times_ns,
                lower[:, channel],
                upper[:, channel],
                where=finite_band,
                color="#CC79A7",
                alpha=0.25,
                label="lower/upper",
            )
            ax.legend(loc="best")
        ax.set_title("Drive intensity")
        ax.set_xlabel(r"$t$ [ns]")
        ax.set_ylabel(r"$I_{14}$ [$10^{14}$ W cm$^{-2}$]")
        ax.grid(True, alpha=0.25)
        if np.any(series["I14"][:, channel] > 1.0e4):
            ax.text(
                0.03,
                0.95,
                "suspect: axis-bin divergence?",
                transform=ax.transAxes,
                va="top",
                ha="left",
                color="#D55E00",
            )

        ax = axes[channel, 3]
        ax.plot(times_ns, series["Ln"][:, channel], color="#E69F00")
        ax.set_title("Evaluation plasma")
        ax.set_xlabel(r"$t$ [ns]")
        ax.set_ylabel(r"$L_n$ [$\mu$m]", color="#E69F00")
        ax.tick_params(axis="y", labelcolor="#E69F00")
        ax.grid(True, alpha=0.25)
        twin = ax.twinx()
        twin.plot(times_ns, series["Te"][:, channel], color="#56B4E9")
        twin.set_ylabel(r"$T_e$ [keV]", color="#56B4E9")
        twin.tick_params(axis="y", labelcolor="#56B4E9")
        tau_ps = series["tau"][:, channel] * 1.0e12
        finite_tau = tau_ps[np.isfinite(tau_ps)]
        tau_mean = float(np.mean(finite_tau)) if finite_tau.size else np.nan
        ax.text(
            0.03,
            0.95,
            rf"$\langle\tau\rangle={tau_mean:.3g}$ ps",
            transform=ax.transAxes,
            va="top",
            ha="left",
            color="black",
        )

    path = out_dir / "eta_channels.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    written(path, used)
    return 1


def port_sky_coordinates(ports):
    axes = ports["axes"]
    theta = np.degrees(np.arccos(np.clip(axes[:, 2], -1.0, 1.0)))
    phi = np.degrees(np.mod(np.arctan2(axes[:, 1], axes[:, 0]), 2.0 * PI))
    return theta, phi


def plot_sky(snapshot, ports, frozen_path, out_dir, dpi):
    name = f"sky_eval_t{time_tag(snapshot['time_s'])}ns.png"
    if ports is None:
        warn_skipped(name, ["config/*_frozen.json:laser.port_configuration.ports"])
        return 0

    try:
        with h5py.File(snapshot["path"], "r") as f:
            values, missing = load_required(f, SKY_PATHS)
    except OSError as exc:
        warn(f"skipping {name}; could not read {snapshot['path']}: {exc}")
        return 0
    if missing:
        warn_skipped(name, missing)
        return 0

    mu = np.asarray(values["mu"], dtype=float).reshape(-1)
    phi = np.mod(np.asarray(values["phi"], dtype=float).reshape(-1), 2.0 * PI)
    maps = [
        np.asarray(values["I_tot"], dtype=float),
        np.asarray(values["I_cw"], dtype=float),
        np.asarray(values["n_sigma"], dtype=float),
    ]
    expected_shape = (mu.size, phi.size)
    if not all(array.shape == expected_shape for array in maps):
        warn(
            f"skipping {name}; sky dataset dimensions do not match "
            f"grid shape {expected_shape}"
        )
        return 0

    theta_deg = np.degrees(np.arccos(np.clip(mu, -1.0, 1.0)))
    phi_deg = np.degrees(phi)
    theta_order = np.argsort(theta_deg)
    phi_order = np.argsort(phi_deg)
    theta_deg = theta_deg[theta_order]
    phi_deg = phi_deg[phi_order]
    maps = [
        array[np.ix_(theta_order, phi_order)]
        for array in maps
    ]
    theta_edges = center_edges(theta_deg, lower=0.0, upper=180.0)
    phi_edges = center_edges(phi_deg, lower=0.0, upper=360.0)
    port_theta, port_phi = port_sky_coordinates(ports)

    fig, axes = plt.subplots(
        1, 3, figsize=(15.0, 5.2), constrained_layout=True
    )
    panels = [
        (maps[0], "viridis", r"$I_{\rm tot}$", r"$I_{\rm tot}$ [W cm$^{-2}$]"),
        (maps[1], "magma", r"$I_{\rm cw}$", r"$I_{\rm cw}$ [W cm$^{-2}$]"),
        (maps[2], "cividis", r"$N_\sigma$", r"$N_\sigma$ [count]"),
    ]
    for ax, (data, cmap, title, colorbar_label) in zip(axes, panels):
        finite_data = data[np.isfinite(data)]
        percentile_vmax = (
            float(np.percentile(finite_data, 99.5))
            if finite_data.size
            else None
        )
        mesh = ax.pcolormesh(
            phi_edges,
            theta_edges,
            data,
            shading="flat",
            cmap=cmap,
            vmax=percentile_vmax,
            rasterized=True,
        )
        ax.scatter(
            port_phi,
            port_theta,
            marker="x",
            s=40,
            linewidths=2.6,
            color="black",
            zorder=3,
        )
        ax.scatter(
            port_phi,
            port_theta,
            marker="x",
            s=40,
            linewidths=1.2,
            color="white",
            zorder=4,
        )
        fig.colorbar(mesh, ax=ax, label=colorbar_label)
        ax.set_title(title)
        ax.set_xlabel(r"$\phi$ [deg]")
        ax.set_ylabel(r"$\theta$ [deg]")
        ax.set_xlim(0.0, 360.0)
        ax.set_ylim(180.0, 0.0)
        ax.set_aspect("equal")

    fig.suptitle(
        rf"Evaluation sphere at $t={snapshot['time_s'] * 1.0e9:.3f}$ ns"
    )
    path = out_dir / name
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    used = list(SKY_PATHS.values()) + [snapshot["time_source"]]
    if frozen_path is not None:
        used.append(
            f"{frozen_path}:laser.port_configuration.ports"
        )
    written(path, used)
    return 1


def critical_density(wavelength_nm):
    wavelength_cm = wavelength_nm * 1.0e-7
    if not wavelength_cm > 0.0:
        return None
    omega = 2.0 * PI * C_LIGHT / wavelength_cm
    return (
        ELECTRON_MASS
        * omega
        * omega
        / (4.0 * PI * ELEMENTARY_CHARGE * ELEMENTARY_CHARGE)
    )


def density_fraction_profile(f, config, n_cells, edges):
    missing = []
    rho = get(f, HYDRO_RHO)
    zbar = get(f, HYDRO_ZBAR)
    volfrac = get(f, HYDRO_VOLFRAC)
    if rho is None:
        missing.append(HYDRO_RHO)
    if zbar is None:
        missing.append(HYDRO_ZBAR)
    if volfrac is None:
        missing.append(HYDRO_VOLFRAC)
    if missing:
        return None, missing, []

    if not isinstance(config, dict):
        return None, ["config/*_frozen.json"], []
    materials_config = config.get("materials")
    laser_config = config.get("laser")
    materials = (
        materials_config.get("materials")
        if isinstance(materials_config, dict)
        else None
    )
    wavelength_nm = (
        laser_config.get("wavelength_nm")
        if isinstance(laser_config, dict)
        else None
    )
    if not isinstance(materials, list) or not materials:
        missing.append("frozen:materials.materials[].A")
    if not isinstance(wavelength_nm, (int, float)):
        missing.append("frozen:laser.wavelength_nm")
    if missing:
        return None, missing, []

    atomic_masses = []
    for material in materials:
        atomic_mass = material.get("A") if isinstance(material, dict) else None
        if not isinstance(atomic_mass, (int, float)) or not atomic_mass > 0.0:
            return None, ["frozen:materials.materials[].A"], []
        atomic_masses.append(float(atomic_mass))

    rho = np.asarray(rho, dtype=float).reshape(-1)
    zbar = np.asarray(zbar, dtype=float).reshape(-1)
    volfrac = np.asarray(volfrac, dtype=float)
    if (
        rho.size != n_cells
        or zbar.size != n_cells
        or volfrac.shape != (n_cells, len(atomic_masses))
    ):
        return None, ["compatible /hydro/rho,zbar,volFrac dimensions"], []

    fractions = np.maximum(volfrac, 0.0)
    fraction_sum = np.sum(fractions, axis=1)
    inverse_a = np.sum(
        fractions / np.asarray(atomic_masses)[np.newaxis, :], axis=1
    )
    valid_fraction = fraction_sum > 1.0e-30
    inverse_a[valid_fraction] /= fraction_sum[valid_fraction]
    a_eff = np.full(n_cells, np.nan)
    valid_a = np.isfinite(inverse_a) & (inverse_a > 1.0e-30)
    a_eff[valid_a] = 1.0 / inverse_a[valid_a]

    n_crit = critical_density(float(wavelength_nm))
    if n_crit is None:
        return None, ["positive frozen:laser.wavelength_nm"], []
    n_hat = rho * np.maximum(zbar, 0.0) / (a_eff * PROTON_MASS * n_crit)
    centers = 0.5 * (edges[:-1] + edges[1:])
    used = [
        HYDRO_RHO,
        HYDRO_ZBAR,
        HYDRO_VOLFRAC,
        "frozen:materials.materials[].A",
        "frozen:laser.wavelength_nm",
    ]
    return (centers, n_hat), [], used


def crossing_radius(centers, n_hat, target):
    for outer in range(len(centers) - 1, 0, -1):
        inner = outer - 1
        n_outer = n_hat[outer]
        n_inner = n_hat[inner]
        if (
            np.isfinite(n_outer)
            and np.isfinite(n_inner)
            and n_outer < target
            and n_inner >= target
            and n_inner != n_outer
        ):
            fraction = (target - n_outer) / (n_inner - n_outer)
            return centers[outer] + fraction * (
                centers[inner] - centers[outer]
            )
    return None


def channel_surface_radii(f, config, n_cells, edges):
    channels = frozen_channels(config)
    if not channels:
        return [], ["frozen:laser.hot_electron.sources"], []
    profile, missing, used = density_fraction_profile(
        f, config, n_cells, edges
    )
    if profile is None:
        return [], missing, used
    centers, n_hat = profile
    surfaces = []
    for channel_index, channel in enumerate(channels):
        if not isinstance(channel, dict):
            continue
        mechanism = channel.get("mechanism", f"ch{channel_index}")
        capture_fraction = channel.get("capture_nc_fraction")
        eval_fraction = channel.get("eval_nc_fraction")
        if isinstance(capture_fraction, (int, float)):
            radius = crossing_radius(centers, n_hat, float(capture_fraction))
            if radius is not None:
                surfaces.append(
                    (
                        f"ch{channel_index} {mechanism} capture",
                        radius,
                        "--",
                    )
                )
        if isinstance(eval_fraction, (int, float)):
            radius = crossing_radius(centers, n_hat, float(eval_fraction))
            if radius is not None:
                surfaces.append(
                    (
                        f"ch{channel_index} {mechanism} eval",
                        radius,
                        ":",
                    )
                )
    return surfaces, [], used


def reconstruct_meridian(ray_map, shell_r, ports, meridian_phi_deg):
    grid_size = 512
    r_max = float(np.max(shell_r))
    limit = 1.1 * r_max
    coordinates = np.linspace(-limit, limit, grid_size)
    R, Z = np.meshgrid(coordinates, coordinates)
    radius = np.hypot(R, Z)
    shell_bin = np.searchsorted(shell_r, radius, side="right") - 1
    shell_valid = (
        (radius > 0.0)
        & (shell_bin >= 0)
        & (shell_bin < ray_map.shape[0])
        & (radius <= r_max)
    )

    phi = np.radians(meridian_phi_deg)
    e_r = np.asarray([np.cos(phi), np.sin(phi), 0.0])
    normal = np.asarray([-np.sin(phi), np.cos(phi), 0.0])
    pixel_direction = np.zeros(R.shape + (3,), dtype=float)
    pixel_direction[..., 0] = R * e_r[0]
    pixel_direction[..., 1] = R * e_r[1]
    pixel_direction[..., 2] = Z
    pixel_direction[shell_valid] /= radius[shell_valid, np.newaxis]

    intensity = np.zeros_like(R)
    selected_ports = []
    band = np.radians(25.0)
    n_theta = ray_map.shape[1]
    for port_index, (axis, weight) in enumerate(
        zip(ports["axes"], ports["weights"])
    ):
        normal_component = float(np.dot(axis, normal))
        plane_angle = float(np.arcsin(np.clip(abs(normal_component), 0.0, 1.0)))
        if plane_angle > band:
            continue
        projected = axis - normal_component * normal
        projected_norm = float(np.linalg.norm(projected))
        if not projected_norm > 0.0:
            continue
        projected /= projected_norm

        # A raised-cosine projection window tapers from one in the plane to
        # zero at +/-25 deg: w_plane = cos^2[(pi/2) delta_phi / 25 deg].
        w_plane = float(np.cos(0.5 * PI * plane_angle / band) ** 2)
        cosine = np.sum(pixel_direction * projected, axis=-1)
        theta_port = np.arccos(np.clip(cosine, -1.0, 1.0))
        theta_bin = np.floor(theta_port * n_theta / PI).astype(int)
        theta_bin = np.clip(theta_bin, 0, n_theta - 1)
        valid_index = np.where(shell_valid)
        samples = np.sum(
            ray_map[
                shell_bin[valid_index],
                theta_bin[valid_index],
                :,
            ],
            axis=1,
        )
        intensity[valid_index] += float(weight) * w_plane * samples
        selected_ports.append((port_index, projected, w_plane))

    return coordinates, intensity, selected_ports, e_r


def plot_meridian(
    snapshot,
    ports,
    config,
    frozen_path,
    meridian_phi_deg,
    out_dir,
    dpi,
):
    name = f"meridian_t{time_tag(snapshot['time_s'])}ns.png"
    if ports is None:
        warn_skipped(name, ["config/*_frozen.json:laser.port_configuration.ports"])
        return 0

    required = {
        "ray_map": RAY_MAP,
        "shell_r": RAY_MAP_SHELL_R,
        "gross": HYDRO_CBET_GROSS,
    }
    try:
        with h5py.File(snapshot["path"], "r") as f:
            values, missing = load_required(f, required)
            if missing:
                warn_skipped(name, missing)
                return 0
            ray_map = np.asarray(values["ray_map"], dtype=float)
            shell_r = np.asarray(values["shell_r"], dtype=float).reshape(-1)
            gross = np.asarray(values["gross"], dtype=float).reshape(-1)
            edges, radius_source, radius_missing = radius_edges(f, gross.size)
            if edges is None:
                warn_skipped(name, radius_missing)
                return 0
            surfaces, surface_missing, surface_used = channel_surface_radii(
                f, config, gross.size, edges
            )
    except OSError as exc:
        warn(f"skipping {name}; could not read {snapshot['path']}: {exc}")
        return 0

    if (
        ray_map.ndim != 3
        or ray_map.shape[2] != 2
        or ray_map.shape[0] != shell_r.size
        or shell_r.size == 0
        or not np.all(np.isfinite(shell_r))
        or not np.all(np.diff(shell_r) > 0.0)
    ):
        warn(f"skipping {name}; incompatible ray_map/ray_map_shell_r dimensions")
        return 0

    coordinates, intensity, selected_ports, e_r = reconstruct_meridian(
        ray_map, shell_r, ports, meridian_phi_deg
    )
    if not selected_ports:
        warn(
            f"{name}: no port axes lie within +/-25 deg of "
            f"phi={meridian_phi_deg:g} deg; reconstructed intensity is zero"
        )
    if surface_missing:
        warn(
            f"{name}: eval/capture circles skipped; missing datasets/config fields: "
            f"{', '.join(surface_missing)}"
        )

    limit_um = float(np.max(np.abs(coordinates))) * 1.0e4
    extent_um = (-limit_um, limit_um, -limit_um, limit_um)
    R, Z = np.meshgrid(coordinates, coordinates)
    radius = np.hypot(R, Z)

    hydro_bin = np.searchsorted(edges, radius, side="right") - 1
    hydro_valid = (
        (hydro_bin >= 0)
        & (hydro_bin < gross.size)
        & (radius <= edges[-1])
    )
    gross_w = gross * ERG_PER_S_TO_W
    gross_max = float(np.nanmax(gross_w)) if gross_w.size else 0.0
    gross_overlay = np.ma.masked_all(radius.shape)
    gross_alpha = np.zeros(radius.shape)
    if np.isfinite(gross_max) and gross_max > 0.0:
        gross_floor = gross_max * 1.0e-8
        radial_values = np.maximum(gross_w[hydro_bin[hydro_valid]], gross_floor)
        normalized = (
            np.log10(radial_values) - np.log10(gross_floor)
        ) / 8.0
        gross_overlay[hydro_valid] = normalized
        gross_alpha[hydro_valid] = 0.30 * normalized

    fig, ax = plt.subplots(figsize=(8.2, 7.2), constrained_layout=True)
    positive_intensity = intensity[
        np.isfinite(intensity) & (intensity > 0.0)
    ]
    if positive_intensity.size:
        intensity_vmax = float(np.percentile(positive_intensity, 99.0))
    else:
        intensity_vmax = 1.0e-29
    intensity_vmin = 1.0e-4 * intensity_vmax
    image = ax.imshow(
        intensity,
        origin="lower",
        extent=extent_um,
        cmap="inferno",
        interpolation="nearest",
        norm=matplotlib.colors.LogNorm(
            vmin=intensity_vmin, vmax=intensity_vmax
        ),
    )
    fig.colorbar(
        image,
        ax=ax,
        label=r"Reconstructed intensity [W cm$^{-2}$] (p99 norm)",
    )
    if np.any(gross_alpha > 0.0):
        ax.imshow(
            gross_overlay,
            origin="lower",
            extent=extent_um,
            cmap="Greys",
            interpolation="nearest",
            alpha=gross_alpha,
            vmin=0.0,
            vmax=1.0,
        )
        ax.text(
            0.02,
            0.02,
            r"grey rings: CBET gross [W cm$^{-3}$], log-scaled alpha",
            transform=ax.transAxes,
            color="white",
            fontsize=8,
            ha="left",
            va="bottom",
        )

    angle_grid = np.linspace(0.0, 2.0 * PI, 721)
    circle_colors = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7"]
    for surface_index, (label, radius_cm, linestyle) in enumerate(surfaces):
        radius_um = radius_cm * 1.0e4
        ax.plot(
            radius_um * np.cos(angle_grid),
            radius_um * np.sin(angle_grid),
            linestyle=linestyle,
            linewidth=1.2,
            color=circle_colors[surface_index % len(circle_colors)],
            label=f"{label}: {radius_um:.1f} um",
        )

    arrow_length_um = 0.88 * float(np.max(shell_r)) * 1.0e4
    for port_index, projected, w_plane in selected_ports:
        projected_r = float(np.dot(projected, e_r))
        projected_z = float(projected[2])
        ax.annotate(
            "",
            xy=(
                arrow_length_um * projected_r,
                arrow_length_um * projected_z,
            ),
            xytext=(0.0, 0.0),
            arrowprops={
                "arrowstyle": "->",
                "color": "#00FFFF",
                "alpha": 0.35 + 0.65 * w_plane,
                "linewidth": 1.0,
            },
        )

    if surfaces:
        ax.legend(loc="upper right", fontsize=7)
    ax.set_xlim(-limit_um, limit_um)
    ax.set_ylim(-limit_um, limit_um)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$R$ [$\mu$m]")
    ax.set_ylabel(r"$Z$ [$\mu$m]")
    ax.set_title(
        rf"$t={snapshot['time_s'] * 1.0e9:.3f}$ ns, "
        rf"$\phi={meridian_phi_deg:g}^\circ$ meridian"
        "\n"
        r"$w_{\rm plane}=\cos^2[(\pi/2)\Delta\phi/25^\circ]$ "
        r"for $|\Delta\phi|\leq25^\circ$"
    )

    path = out_dir / name
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    used = list(required.values()) + [radius_source, snapshot["time_source"]]
    used.extend(surface_used)
    if frozen_path is not None:
        used.append(
            f"{frozen_path}:laser.port_configuration.ports"
        )
        if surfaces:
            used.append(
                f"{frozen_path}:laser.hot_electron.sources"
            )
    written(path, used)
    return 1


def plot_ports(snapshots, ports, frozen_path, out_dir, dpi):
    required = {
        "outgoing": PORT_OUTGOING_POWER,
        "capture": PORT_CAPTURE_PCROSS,
    }
    rows = []
    missing_all = []
    used = []
    n_ports = None

    for snapshot in snapshots:
        try:
            with h5py.File(snapshot["path"], "r") as f:
                values, missing = load_required(f, required)
        except OSError as exc:
            warn(f"ports.png: omitted unreadable {snapshot['path']}: {exc}")
            continue
        if missing:
            missing_all.extend(missing)
            warn(
                f"ports.png: omitted {snapshot['path'].name}; "
                f"missing datasets: {', '.join(missing)}"
            )
            continue

        outgoing = np.asarray(values["outgoing"], dtype=float).reshape(-1)
        capture = np.asarray(values["capture"], dtype=float)
        if capture.ndim != 2 or capture.shape[0] != outgoing.size:
            warn(
                f"ports.png: omitted {snapshot['path'].name}; "
                "port_capture_Pcross dimensions do not match port_outgoing_power"
            )
            continue
        if n_ports is None:
            n_ports = outgoing.size
        if outgoing.size != n_ports:
            warn(
                f"ports.png: omitted {snapshot['path'].name}; "
                f"port count {outgoing.size} differs from {n_ports}"
            )
            continue
        rows.append(
            {
                "time_ns": snapshot["time_s"] * 1.0e9,
                "outgoing": outgoing * ERG_PER_S_TO_W,
                "capture": np.sum(capture, axis=1) * ERG_PER_S_TO_W,
            }
        )
        used.extend(required.values())
        used.append(snapshot["time_source"])

    if not rows or not n_ports:
        warn_skipped("ports.png", missing_all or list(required.values()))
        return 0

    times_ns = np.asarray([row["time_ns"] for row in rows])
    time_edges = center_edges(times_ns)
    port_edges = np.arange(n_ports + 1, dtype=float) - 0.5
    outgoing = np.asarray([row["outgoing"] for row in rows])
    capture = np.asarray([row["capture"] for row in rows])

    if ports is not None and len(ports["ids"]) == n_ports:
        port_labels = [str(port_id) for port_id in ports["ids"]]
        if frozen_path is not None:
            used.append(f"{frozen_path}:laser.port_configuration.ports[].port_id")
    else:
        port_labels = [str(index) for index in range(n_ports)]

    fig, axes = plt.subplots(
        1, 3, figsize=(16.0, 4.8), constrained_layout=True
    )
    outgoing_mesh = axes[0].pcolormesh(
        time_edges,
        port_edges,
        outgoing.T,
        cmap="viridis",
        shading="flat",
        rasterized=True,
    )
    fig.colorbar(outgoing_mesh, ax=axes[0], label="Outgoing power [W]")
    axes[0].set_title("Port outgoing power")
    axes[0].set_xlabel(r"$t$ [ns]")
    axes[0].set_ylabel("Port ID [1]")
    axes[0].set_yticks(np.arange(n_ports))
    axes[0].set_yticklabels(port_labels)

    capture_mesh = axes[1].pcolormesh(
        time_edges,
        port_edges,
        capture.T,
        cmap="magma",
        shading="flat",
        rasterized=True,
    )
    fig.colorbar(
        capture_mesh,
        ax=axes[1],
        label=r"$\sum_{\rm channels}P_{\rm cross}$ [W]",
    )
    axes[1].set_title("Port capture crossing power")
    axes[1].set_xlabel(r"$t$ [ns]")
    axes[1].set_ylabel("Port ID [1]")
    axes[1].set_yticks(np.arange(n_ports))
    axes[1].set_yticklabels(port_labels)

    positions = np.arange(n_ports, dtype=float)
    width = 0.38
    axes[2].bar(
        positions - 0.5 * width,
        outgoing[-1],
        width=width,
        color="#0072B2",
        label="outgoing",
    )
    axes[2].bar(
        positions + 0.5 * width,
        capture[-1],
        width=width,
        color="#D55E00",
        label=r"$\sum P_{\rm cross}$",
    )
    axes[2].set_title(rf"Final snapshot: $t={times_ns[-1]:.3f}$ ns")
    axes[2].set_xlabel("Port ID [1]")
    axes[2].set_ylabel("Power [W]")
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(port_labels, rotation=45, ha="right")
    axes[2].legend(loc="best")
    axes[2].grid(True, axis="y", alpha=0.25)

    path = out_dir / "ports.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    written(path, used)
    return 1


def parse_times(text):
    if text is None:
        return None
    values = []
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            raise argparse.ArgumentTypeError(
                "--times must be a comma-separated list of times in ns"
            )
        try:
            value = float(stripped)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid time in --times: {stripped}"
            ) from exc
        if not np.isfinite(value):
            raise argparse.ArgumentTypeError(
                f"non-finite time in --times: {stripped}"
            )
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("--times must not be empty")
    return values


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot multibeam CBET/hot-electron diagnostics."
    )
    parser.add_argument(
        "RUN_DIR",
        type=pathlib.Path,
        help="TENRYU output directory containing results/*.h5",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help="output directory for PNG files",
    )
    parser.add_argument(
        "--times",
        type=parse_times,
        default=None,
        help="comma-separated snapshot times in ns",
    )
    parser.add_argument(
        "--meridian-phi",
        type=float,
        default=0.0,
        metavar="PHI_DEG",
        help="meridian-plane azimuth in degrees (default: 0)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PNG resolution in dots per inch (default: 150)",
    )
    return parser


def main():
    args = build_parser().parse_args()
    run_dir = args.RUN_DIR.expanduser()
    if not run_dir.is_dir():
        warn(f"RUN_DIR is not a directory: {run_dir}")
        return 2
    if not np.isfinite(args.meridian_phi):
        warn("--meridian-phi must be finite")
        return 2
    if args.dpi <= 0:
        warn("--dpi must be positive")
        return 2

    if args.out is None:
        out_dir = (
            pathlib.Path("~/work/projects/TENRYU/outputs/plots")
            .expanduser()
            / run_dir.resolve().name
        )
    else:
        out_dir = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshots = discover_snapshots(run_dir)
    if not snapshots:
        warn_skipped("all figures", [str(run_dir / "results" / "*.h5")])
        return 2

    config, frozen_path = load_frozen_config(run_dir)
    ports = normalized_ports(config)
    selected = selected_snapshots(snapshots, args.times)

    figure_count = 0
    figure_count += plot_streaks(snapshots, out_dir, args.dpi)
    figure_count += plot_eta_channels(snapshots, config, out_dir, args.dpi)
    for snapshot in selected:
        figure_count += plot_sky(
            snapshot, ports, frozen_path, out_dir, args.dpi
        )
        figure_count += plot_meridian(
            snapshot,
            ports,
            config,
            frozen_path,
            args.meridian_phi,
            out_dir,
            args.dpi,
        )
    figure_count += plot_ports(
        snapshots, ports, frozen_path, out_dir, args.dpi
    )
    return 0 if figure_count else 2


if __name__ == "__main__":
    raise SystemExit(main())
