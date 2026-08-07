#!/usr/bin/env python3
"""Create literature-style multibeam summary figures from a TENRYU run."""

import argparse
import json
import pathlib

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge


plt.rcParams["font.family"] = [
    "Hiragino Sans",
    "Hiragino Kaku Gothic ProN",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False


HYDRO_RHO = "/hydro/rho"
HYDRO_TE = "/hydro/Te"
HYDRO_TI = "/hydro/Ti"
HYDRO_VOL = "/hydro/vol"
HYDRO_ZBAR = "/hydro/zbar"
HYDRO_VOLFRAC = "/hydro/volFrac"
HYDRO_CBET_GROSS = "/hydro/cbet_gross_exchange"
HYDRO_HOT_E_Q = "/hydro/hot_e_Q"
HYDRO_ETA = "/hydro/hot_e_eta_model_eta"
HYDRO_G = "/hydro/hot_e_eta_model_g"
LASER_DEPOSITED_POWER = "/laser/deposited_power"
# hdf5_writer.cpp exports deposited_power as dep/(vol*dt), in erg/cm3/s.
LASER_N_CRIT = "/laser/mesh/n_crit"
MESH_X_R = "/mesh/x_r"
MESH_V_R = "/mesh/v_r"
PORT_SECTION = "/laser/port_section"
SKY_MU = f"{PORT_SECTION}/grid_mu"
SKY_PHI = f"{PORT_SECTION}/grid_phi"
SKY_I_TOT = f"{PORT_SECTION}/I_tot_eval"
SKY_I_CW = f"{PORT_SECTION}/I_cw_eval"

PI = np.pi
C_LIGHT = 2.99792458e10
ELECTRON_MASS = 9.1094e-28
ELEMENTARY_CHARGE = 4.8032e-10
PROTON_MASS = 1.6726219e-24
EV_TO_ERG = 1.602e-12
ERG_PER_S_TO_W = 1.0e-7
ERG_PER_S_TO_TW = 1.0e-19
W_TO_TW = 1.0e-12
CM_TO_UM = 1.0e4


def get(handle, path):
    """Return an HDF5 dataset as an array, or None when it is absent."""
    try:
        return np.asarray(handle[path][...])
    except (KeyError, OSError, ValueError):
        return None


def warn(message):
    print(f"WARNING: {message}")


def warn_skipped(figure_name, missing):
    missing_text = ", ".join(sorted(set(missing)))
    warn(f"skipping {figure_name}; missing datasets/config: {missing_text}")


def written(path, datasets):
    used_text = ", ".join(sorted(set(str(item) for item in datasets if item)))
    print(f"WROTE {path} | datasets: {used_text}")


def snapshot_time(handle):
    if "t" in handle.attrs:
        value = np.asarray(handle.attrs["t"]).reshape(-1)
        if value.size == 1 and np.isfinite(value[0]):
            return float(value[0]), "@t"
    value = get(handle, "/time_state/t")
    if value is not None:
        flat = np.asarray(value).reshape(-1)
        if flat.size == 1 and np.isfinite(flat[0]):
            return float(flat[0]), "/time_state/t"
    return None, None


def discover_snapshots(run_dir):
    snapshots = []
    for path in sorted((run_dir / "results").glob("*.h5")):
        try:
            with h5py.File(path, "r") as handle:
                time_s, time_source = snapshot_time(handle)
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


def load_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        warn(f"could not read JSON {path}: {exc}")
        return None


def load_frozen_config(run_dir):
    paths = sorted((run_dir / "config").glob("*_frozen.json"))
    if not paths:
        warn(f"no frozen config found under {run_dir / 'config'}")
        return None, None
    if len(paths) > 1:
        warn(f"multiple frozen configs found; using deterministic first path {paths[0]}")
    return load_json(paths[0]), paths[0]


def load_run_info(run_dir):
    path = run_dir / "run_info.json"
    if not path.is_file():
        return None, path
    return load_json(path), path


def normalized_ports(config):
    if not isinstance(config, dict):
        return None
    laser = config.get("laser")
    port_configuration = (
        laser.get("port_configuration") if isinstance(laser, dict) else None
    )
    port_table = (
        port_configuration.get("ports")
        if isinstance(port_configuration, dict)
        else None
    )
    if not isinstance(port_table, list) or not port_table:
        return None

    axes = []
    weights = []
    ids = []
    delta_lambda = []
    for index, port in enumerate(port_table):
        if not isinstance(port, dict):
            return None
        direction = np.asarray(port.get("direction", []), dtype=float)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            return None
        norm = float(np.linalg.norm(direction))
        weight = port.get("power_weight")
        if (
            not norm > 0.0
            or not isinstance(weight, (int, float))
            or not np.isfinite(weight)
        ):
            return None
        delta = port.get("delta_lambda_nm", 0.0)
        if not isinstance(delta, (int, float)) or not np.isfinite(delta):
            return None
        axes.append(direction / norm)
        weights.append(float(weight))
        ids.append(port.get("port_id", index))
        delta_lambda.append(float(delta))
    return {
        "axes": np.asarray(axes),
        "weights": np.asarray(weights),
        "ids": ids,
        "delta_lambda": np.asarray(delta_lambda),
    }


def frozen_channels(config):
    if not isinstance(config, dict):
        return []
    laser = config.get("laser")
    hot_electron = laser.get("hot_electron") if isinstance(laser, dict) else None
    sources = hot_electron.get("sources") if isinstance(hot_electron, dict) else None
    return sources if isinstance(sources, list) else []


def radius_edges(handle, n_cells):
    x_r = get(handle, MESH_X_R)
    if x_r is not None:
        edges = np.asarray(x_r, dtype=float).reshape(-1)
        if (
            edges.size == n_cells + 1
            and np.all(np.isfinite(edges))
            and np.all(np.diff(edges) > 0.0)
        ):
            return edges, MESH_X_R, []

    geometry = handle.attrs.get("geometry")
    if isinstance(geometry, bytes):
        geometry = geometry.decode(errors="replace")
    vol = get(handle, HYDRO_VOL)
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


def critical_density(wavelength_nm):
    """Return n_c from the standard Gaussian-cgs cold-plasma formula.

    The source formula is n_c = m_e omega^2/(4*pi*e^2), with
    omega = 2*pi*c/lambda.
    """
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


def material_atomic_mass_profile(config, volfrac, n_cells):
    if not isinstance(config, dict):
        return None, ["config/*_frozen.json"]
    materials_block = config.get("materials")
    materials = (
        materials_block.get("materials")
        if isinstance(materials_block, dict)
        else None
    )
    if not isinstance(materials, list) or not materials:
        return None, ["frozen:materials.materials[].A"]

    atomic_masses = []
    for material in materials:
        atomic_mass = material.get("A") if isinstance(material, dict) else None
        if (
            not isinstance(atomic_mass, (int, float))
            or not np.isfinite(atomic_mass)
            or not atomic_mass > 0.0
        ):
            return None, ["frozen:materials.materials[].A"]
        atomic_masses.append(float(atomic_mass))

    if len(atomic_masses) == 1:
        return np.full(n_cells, atomic_masses[0]), []
    if volfrac is None:
        return None, [HYDRO_VOLFRAC]

    fractions = np.asarray(volfrac, dtype=float)
    if fractions.shape != (n_cells, len(atomic_masses)):
        return None, ["compatible /hydro/volFrac dimensions"]
    fractions = np.maximum(fractions, 0.0)
    fraction_sum = np.sum(fractions, axis=1)
    inverse_a = np.sum(
        fractions / np.asarray(atomic_masses)[np.newaxis, :], axis=1
    )
    valid_fraction = fraction_sum > 1.0e-30
    inverse_a[valid_fraction] /= fraction_sum[valid_fraction]
    a_eff = np.full(n_cells, np.nan)
    valid_a = np.isfinite(inverse_a) & (inverse_a > 1.0e-30)
    a_eff[valid_a] = 1.0 / inverse_a[valid_a]
    return a_eff, []


def radial_plasma_state(handle, config):
    required = {
        "rho": HYDRO_RHO,
        "zbar": HYDRO_ZBAR,
        "Te": HYDRO_TE,
        "Ti": HYDRO_TI,
        "v_r": MESH_V_R,
    }
    values = {}
    missing = []
    for key, path in required.items():
        value = get(handle, path)
        if value is None:
            missing.append(path)
        else:
            values[key] = np.asarray(value, dtype=float).reshape(-1)
    if missing:
        return None, missing, []

    n_cells = values["rho"].size
    if n_cells == 0 or any(
        values[key].size != n_cells for key in ("zbar", "Te", "Ti")
    ):
        return None, ["compatible hydro profile dimensions"], []
    if values["v_r"].size != n_cells + 1:
        return None, [f"{MESH_V_R} with n_cells+1 nodes"], []

    edges, radius_source, radius_missing = radius_edges(handle, n_cells)
    if edges is None:
        return None, radius_missing, []
    centers = 0.5 * (edges[:-1] + edges[1:])

    volfrac = get(handle, HYDRO_VOLFRAC)
    a_eff, a_missing = material_atomic_mass_profile(config, volfrac, n_cells)
    if a_eff is None:
        return None, a_missing, []

    n_crit_value = get(handle, LASER_N_CRIT)
    n_crit_source = LASER_N_CRIT
    if n_crit_value is not None:
        flat = np.asarray(n_crit_value, dtype=float).reshape(-1)
        n_crit = float(flat[0]) if flat.size == 1 else np.nan
    else:
        laser = config.get("laser") if isinstance(config, dict) else None
        wavelength = laser.get("wavelength_nm") if isinstance(laser, dict) else None
        n_crit = (
            critical_density(float(wavelength))
            if isinstance(wavelength, (int, float))
            else None
        )
        n_crit_source = (
            "frozen:laser.wavelength_nm -> "
            "n_c=m_e*omega^2/(4*pi*e^2)"
        )
    if n_crit is None or not np.isfinite(n_crit) or not n_crit > 0.0:
        return None, [LASER_N_CRIT, "positive frozen:laser.wavelength_nm"], []

    rho = values["rho"]
    zbar = np.maximum(values["zbar"], 0.0)
    n_hat = rho * zbar / (a_eff * PROTON_MASS * n_crit)
    sound_numerator = (zbar * values["Te"] + 3.0 * values["Ti"]) * EV_TO_ERG
    sound_speed = np.full(n_cells, np.nan)
    valid_sound = (
        np.isfinite(sound_numerator)
        & (sound_numerator > 0.0)
        & np.isfinite(a_eff)
        & (a_eff > 0.0)
    )
    sound_speed[valid_sound] = np.sqrt(
        sound_numerator[valid_sound] / (a_eff[valid_sound] * PROTON_MASS)
    )
    velocity = 0.5 * (values["v_r"][:-1] + values["v_r"][1:])
    mach = np.divide(
        velocity,
        sound_speed,
        out=np.full(n_cells, np.nan),
        where=np.isfinite(sound_speed) & (sound_speed > 0.0),
    )

    used = list(required.values()) + [
        radius_source,
        "frozen:materials.materials[].A",
        n_crit_source,
    ]
    if len(np.unique(a_eff[np.isfinite(a_eff)])) > 1:
        used.append(HYDRO_VOLFRAC)
    return {
        "n_cells": n_cells,
        "edges": edges,
        "centers": centers,
        "rho": rho,
        "n_hat": n_hat,
        "mach": mach,
        "n_crit": n_crit,
    }, [], used


def outermost_crossing_radius(centers, values, target):
    centers = np.asarray(centers, dtype=float)
    values = np.asarray(values, dtype=float)
    for outer in range(values.size - 1, 0, -1):
        inner = outer - 1
        y0 = values[inner]
        y1 = values[outer]
        if not np.isfinite(y0) or not np.isfinite(y1):
            continue
        if y1 == target:
            return float(centers[outer])
        if y0 == target:
            return float(centers[inner])
        if (y0 - target) * (y1 - target) < 0.0:
            fraction = (target - y0) / (y1 - y0)
            return float(centers[inner] + fraction * (centers[outer] - centers[inner]))
    return None


def cbet_transfer_band(edges, gross):
    gross = np.asarray(gross, dtype=float).reshape(-1)
    finite = gross[np.isfinite(gross)]
    maximum = float(np.max(finite)) if finite.size else 0.0
    if not maximum > 0.0:
        return None
    indices = np.flatnonzero(np.isfinite(gross) & (gross > 0.1 * maximum))
    if not indices.size:
        return None
    return float(edges[indices[0]]), float(edges[indices[-1] + 1])


def target_radius(centers, rho):
    dense = np.flatnonzero(np.isfinite(rho) & (rho > 1.0))
    if dense.size:
        return float(centers[dense[-1]]), "rho > 1 g/cc", False
    finite = rho[np.isfinite(rho)]
    maximum = float(np.max(finite)) if finite.size else 0.0
    fallback = np.flatnonzero(np.isfinite(rho) & (rho > 0.1 * maximum))
    if fallback.size and maximum > 0.0:
        return float(centers[fallback[-1]]), "rho > 0.1*max(rho)", True
    return None, None, True


def channel_capture_surfaces(config, centers, n_hat):
    surfaces = []
    for index, channel in enumerate(frozen_channels(config)):
        if not isinstance(channel, dict):
            continue
        fraction = channel.get("capture_nc_fraction")
        if not isinstance(fraction, (int, float)) or not np.isfinite(fraction):
            continue
        radius = outermost_crossing_radius(centers, n_hat, float(fraction))
        if radius is None:
            continue
        mechanism = str(channel.get("mechanism", f"ch{index}")).upper()
        surfaces.append(
            {
                "index": index,
                "mechanism": mechanism,
                "fraction": float(fraction),
                "radius": radius,
            }
        )
    return surfaces


def selected_snapshot(snapshots, requested_time_ns):
    if requested_time_ns is None:
        target_s = 0.75 * snapshots[-1]["time_s"]
    else:
        target_s = requested_time_ns * 1.0e-9
    return min(snapshots, key=lambda item: abs(item["time_s"] - target_s))


def add_surface_annotation(ax, text, radius, angle_deg, text_xy):
    angle = np.radians(angle_deg)
    point = (radius * np.cos(angle), radius * np.sin(angle))
    ax.annotate(
        text,
        xy=point,
        xytext=text_xy,
        ha="left" if text_xy[0] >= 0.0 else "right",
        va="center",
        fontsize=9,
        arrowprops={
            "arrowstyle": "-",
            "color": "#333333",
            "linewidth": 0.9,
        },
        zorder=12,
    )


def normalized_curve(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    peak = float(np.max(finite)) if finite.size else 0.0
    if not peak > 0.0:
        return None, peak
    return np.where(np.isfinite(values), values / peak, np.nan), peak


def plot_fig1(snapshot, config, ports, frozen_path, out_dir, dpi):
    name = "fig1_overview.png"
    if ports is None:
        warn_skipped(name, ["frozen:laser.port_configuration.ports"])
        return 0

    try:
        with h5py.File(snapshot["path"], "r") as handle:
            plasma, missing, used = radial_plasma_state(handle, config)
            if plasma is None:
                warn_skipped(name, missing)
                return 0
            gross_value = get(handle, HYDRO_CBET_GROSS)
            if gross_value is None:
                warn_skipped(name, [HYDRO_CBET_GROSS])
                return 0
            gross = np.asarray(gross_value, dtype=float).reshape(-1)
            if gross.size != plasma["n_cells"]:
                warn_skipped(name, ["compatible cbet_gross_exchange dimensions"])
                return 0
            profile_values = {
                "IB 吸収密度": get(handle, LASER_DEPOSITED_POWER),
                "CBET 交換密度": gross,
                "hot-e 沈着密度 (輸送後)": get(handle, HYDRO_HOT_E_Q),
            }
    except OSError as exc:
        warn(f"skipping {name}; could not read {snapshot['path']}: {exc}")
        return 0

    edges = plasma["edges"]
    centers = plasma["centers"]
    band = cbet_transfer_band(edges, gross)
    if band is None:
        warn(f"{name}: CBET transfer band unavailable because gross exchange is non-positive")

    r_target, target_source, used_fallback = target_radius(centers, plasma["rho"])
    if r_target is None:
        warn_skipped(name, ["positive ablation-surface density threshold"])
        return 0
    if used_fallback:
        warn(f"{name}: ablation surface used fallback {target_source}")

    r_nc = outermost_crossing_radius(centers, plasma["n_hat"], 1.0)
    r_quarter = outermost_crossing_radius(centers, plasma["n_hat"], 0.25)
    r_mach = outermost_crossing_radius(centers, plasma["mach"], 1.0)
    profile_xlim_um = (
        1.3 * r_quarter * CM_TO_UM if r_quarter is not None else 300.0
    )
    if r_nc is None:
        warn(f"{name}: n_c surface not crossed in the selected hydro profile")
    if r_quarter is None:
        warn(f"{name}: n_c/4 surface not crossed in the selected hydro profile")
    if r_mach is None:
        warn(f"{name}: Mach-1 surface not crossed in the selected hydro profile")

    capture_surfaces = channel_capture_surfaces(
        config, centers, plasma["n_hat"]
    )
    if not capture_surfaces:
        warn(f"{name}: no hot-e channel capture surface crossed the hydro profile")

    physical_radii = [r_target]
    physical_radii.extend(
        radius for radius in (r_nc, r_quarter, r_mach) if radius is not None
    )
    physical_radii.extend(surface["radius"] for surface in capture_surfaces)
    if band is not None:
        physical_radii.extend(band)
    scale = max(physical_radii)

    fig = plt.figure(figsize=(16.0, 7.4), constrained_layout=True)
    outer_grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.2))
    schematic = fig.add_subplot(outer_grid[0, 0])
    profile_grid = outer_grid[0, 1].subgridspec(
        2, 1, height_ratios=(1.0, 1.0), hspace=0.08
    )
    top = fig.add_subplot(profile_grid[0, 0])
    bottom = fig.add_subplot(profile_grid[1, 0], sharex=top)

    schematic.add_patch(
        Circle(
            (0.0, 0.0),
            r_target,
            facecolor="#B8E0B8",
            edgecolor="#3A7D44",
            linewidth=1.2,
            zorder=1,
        )
    )
    if band is not None:
        r_inner, r_outer = band
        schematic.add_patch(
            Wedge(
                (0.0, 0.0),
                r_outer,
                0.0,
                360.0,
                width=max(r_outer - r_inner, 1.0e-30),
                facecolor="#D62728",
                edgecolor="none",
                alpha=0.22,
                zorder=2,
            )
        )
    if r_nc is not None:
        schematic.add_patch(
            Circle(
                (0.0, 0.0),
                r_nc,
                fill=False,
                edgecolor="black",
                linewidth=1.8,
                zorder=5,
            )
        )
    if r_quarter is not None:
        schematic.add_patch(
            Circle(
                (0.0, 0.0),
                r_quarter,
                fill=False,
                edgecolor="#555555",
                linestyle="--",
                linewidth=1.6,
                zorder=5,
            )
        )
    if r_mach is not None:
        schematic.add_patch(
            Circle(
                (0.0, 0.0),
                r_mach,
                fill=False,
                edgecolor="#1F77B4",
                linestyle="-.",
                linewidth=1.6,
                zorder=5,
            )
        )
    for surface in capture_surfaces:
        schematic.add_patch(
            Circle(
                (0.0, 0.0),
                surface["radius"],
                fill=False,
                edgecolor="#E67E22",
                linestyle=":",
                linewidth=1.4,
                zorder=6,
            )
        )

    beam_outer = 1.33 * scale
    beam_inner = 1.03 * scale
    for direction in ports["axes"]:
        projected = direction[[0, 2]]
        projected_norm = float(np.linalg.norm(projected))
        if not projected_norm > 0.0:
            continue
        projected = projected / projected_norm
        schematic.annotate(
            "",
            xy=beam_inner * projected,
            xytext=beam_outer * projected,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#00AFC8",
                "linewidth": 1.35,
                "alpha": 0.82,
            },
            zorder=8,
        )

    flow_outer = max(0.93 * scale, 1.22 * r_target)
    for angle in np.linspace(0.0, 2.0 * PI, 6, endpoint=False) + PI / 12.0:
        direction = np.asarray([np.cos(angle), np.sin(angle)])
        schematic.annotate(
            "",
            xy=flow_outer * direction,
            xytext=1.03 * r_target * direction,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#2CA02C",
                "linewidth": 1.8,
                "alpha": 0.78,
            },
            zorder=7,
        )

    label_extent = 1.58 * scale
    add_surface_annotation(
        schematic,
        "アブレーション面",
        r_target,
        215.0,
        (-1.03 * label_extent, -0.55 * label_extent),
    )
    if r_nc is not None:
        add_surface_annotation(
            schematic,
            "臨界面 n_c",
            r_nc,
            35.0,
            (0.62 * label_extent, 0.56 * label_extent),
        )
    if r_quarter is not None:
        add_surface_annotation(
            schematic,
            "1/4 臨界 (TPD/捕獲面)",
            r_quarter,
            145.0,
            (-0.46 * label_extent, 0.77 * label_extent),
        )
    if r_mach is not None:
        add_surface_annotation(
            schematic,
            "Mach 1 (CBET 転送帯)",
            r_mach,
            300.0,
            (0.44 * label_extent, -0.77 * label_extent),
        )
    if band is not None:
        add_surface_annotation(
            schematic,
            "CBET 転送帯",
            0.5 * (band[0] + band[1]),
            190.0,
            (-0.70 * label_extent, -0.22 * label_extent),
        )
    for surface_index, surface in enumerate(capture_surfaces):
        add_surface_annotation(
            schematic,
            f"hot-e 捕獲面 ({surface['mechanism']})",
            surface["radius"],
            70.0 + 18.0 * surface_index,
            (
                0.54 * label_extent,
                (0.27 - 0.13 * surface_index) * label_extent,
            ),
        )
    schematic.annotate(
        "入射ビーム (12 港)",
        xy=(1.18 * scale, 0.44 * scale),
        xytext=(0.60 * label_extent, 0.92 * label_extent),
        ha="left",
        va="center",
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": "#00AFC8", "linewidth": 1.0},
    )
    schematic.annotate(
        "外向きコロナ流",
        xy=(0.65 * scale, 0.17 * scale),
        xytext=(0.73 * label_extent, -0.10 * label_extent),
        ha="left",
        va="center",
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": "#2CA02C", "linewidth": 1.0},
    )
    schematic.set_xlim(-label_extent, label_extent)
    schematic.set_ylim(-label_extent, label_extent)
    schematic.set_aspect("equal")
    schematic.axis("off")
    schematic.set_title(
        f"(a) 模式図  t = {snapshot['time_s'] * 1.0e9:.3f} ns",
        loc="left",
    )

    r_um = centers * CM_TO_UM
    curve_specs = [
        ("IB 吸収密度", "black", LASER_DEPOSITED_POWER),
        ("CBET 交換密度", "#1F77B4", HYDRO_CBET_GROSS),
        ("hot-e 沈着密度 (輸送後)", "#D62728", HYDRO_HOT_E_Q),
    ]
    plotted_profiles = 0
    for label, color, source_path in curve_specs:
        value = profile_values[label]
        if value is None:
            warn(f"{name}: omitted {label}; missing dataset {source_path}")
            continue
        density = np.asarray(value, dtype=float).reshape(-1)
        if density.size != plasma["n_cells"]:
            warn(f"{name}: omitted {label}; incompatible profile dimensions")
            continue
        density_w = density * ERG_PER_S_TO_W
        normalized, peak = normalized_curve(density_w)
        if normalized is None:
            warn(f"{name}: omitted {label}; profile maximum is non-positive")
            continue
        top.plot(
            r_um,
            normalized,
            color=color,
            linewidth=1.7,
            label=f"{label} (最大 {peak:.2e} W/cm³)",
        )
        used.append(source_path)
        plotted_profiles += 1
    for surface in capture_surfaces:
        capture_radius_um = surface["radius"] * CM_TO_UM
        top.axvline(
            capture_radius_um,
            color="#E67E22",
            linestyle="--",
            linewidth=1.2,
        )
        top.text(
            capture_radius_um,
            0.97,
            f"生成面 ({surface['mechanism']})",
            color="#E67E22",
            fontsize=8,
            rotation=90,
            ha="right",
            va="top",
            transform=top.get_xaxis_transform(),
        )
    if plotted_profiles:
        top.legend(loc="best", fontsize=8)
    top.set_ylabel("(規格化)")
    top.set_ylim(bottom=0.0)
    top.set_xlim(0.0, profile_xlim_um)
    top.grid(True, alpha=0.25)
    top.tick_params(labelbottom=False)
    top.set_title("(b) 半径プロファイル", loc="left")

    bottom.plot(
        r_um,
        plasma["n_hat"],
        color="black",
        linewidth=1.7,
        label="n_e/n_c",
    )
    bottom.axhline(0.25, color="#777777", linestyle="--", linewidth=1.0)
    bottom.axhline(1.0, color="#777777", linestyle="--", linewidth=1.0)
    bottom.set_ylabel("n_e/n_c")
    bottom.set_xlabel("半径 r [um]")
    bottom.set_ylim(0.0, 2.0)
    bottom.set_xlim(0.0, profile_xlim_um)
    bottom.grid(True, alpha=0.25)
    finite_n_hat = plasma["n_hat"][np.isfinite(plasma["n_hat"])]
    if finite_n_hat.size and float(np.max(finite_n_hat)) > 2.0:
        peak_index = int(np.nanargmax(plasma["n_hat"]))
        bottom.annotate(
            "殻ピーク (域外)",
            xy=(r_um[peak_index], 1.98),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#333333",
                "linewidth": 0.8,
            },
        )
    mach_axis = bottom.twinx()
    mach_axis.plot(
        r_um,
        plasma["mach"],
        color="#1F77B4",
        linestyle="--",
        linewidth=1.5,
        label="Mach u_r/c_a",
    )
    mach_axis.axhline(1.0, color="#1F77B4", linestyle=":", linewidth=1.0)
    mach_axis.set_ylabel("Mach u_r/c_a", color="#1F77B4")
    mach_axis.set_ylim(-2.0, 4.0)
    mach_axis.tick_params(axis="y", labelcolor="#1F77B4")
    lines = bottom.get_lines()[:1] + mach_axis.get_lines()[:1]
    bottom.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=8)

    if band is not None:
        for axis in (top, bottom):
            axis.axvspan(
                band[0] * CM_TO_UM,
                band[1] * CM_TO_UM,
                color="#D62728",
                alpha=0.14,
                zorder=0,
            )
        top.text(
            0.5 * (band[0] + band[1]) * CM_TO_UM,
            0.96,
            "転送帯",
            ha="center",
            va="top",
            color="#A61C1C",
            fontsize=9,
        )

    path = out_dir / name
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    used.extend(
        [
            HYDRO_CBET_GROSS,
            snapshot["time_source"],
            f"{frozen_path}:laser.port_configuration.ports" if frozen_path else None,
            f"{frozen_path}:laser.hot_electron.sources" if frozen_path else None,
        ]
    )
    written(path, used)
    return 1


def iter_json(node, path=""):
    yield path, node
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}" if path else str(key)
            yield from iter_json(value, child)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            child = f"{path}[{index}]"
            yield from iter_json(value, child)


def numeric_vector(value):
    if not isinstance(value, list):
        return None
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size < 2 or not np.all(np.isfinite(array)):
        return None
    return array


def table_arrays(node):
    if not isinstance(node, dict):
        return None
    pairs = (
        ("x", "y"),
        ("t", "y"),
        ("times", "values"),
        ("time", "power"),
        ("time_s", "power_W"),
        ("times_s", "power_W"),
        ("t_s", "P_W"),
    )
    for x_key, y_key in pairs:
        x_values = numeric_vector(node.get(x_key))
        y_values = numeric_vector(node.get(y_key))
        if (
            x_values is not None
            and y_values is not None
            and x_values.size == y_values.size
            and np.all(np.diff(x_values) > 0.0)
        ):
            units = str(node.get("units", node.get("y_units", "W")))
            return x_values, y_values, units
    for key in ("table", "samples", "points"):
        value = node.get(key)
        if not isinstance(value, list):
            continue
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            continue
        if (
            array.ndim == 2
            and array.shape[0] >= 2
            and array.shape[1] == 2
            and np.all(np.isfinite(array))
            and np.all(np.diff(array[:, 0]) > 0.0)
        ):
            units = str(node.get("units", node.get("y_units", "W")))
            return array[:, 0], array[:, 1], units
    return None


def find_power_tables(payload):
    tables = []
    for path, node in iter_json(payload):
        lowered = path.lower()
        if "laser" not in lowered or (
            "waveform" not in lowered and "power" not in lowered
        ):
            continue
        arrays = table_arrays(node)
        if arrays is not None:
            tables.append((path, *arrays))
    return tables


def summary_waveform_paths(config):
    paths = []
    for path, node in iter_json(config):
        lowered = path.lower()
        if (
            isinstance(node, dict)
            and "waveform" in lowered
            and {"t_min", "t_max", "n_points", "peak_value"}.issubset(node)
        ):
            paths.append(path)
    return paths


def incident_power_series(config, run_info, times_s, frozen_path, run_info_path):
    candidates = find_power_tables(config)
    source_root = frozen_path
    if not candidates:
        candidates = find_power_tables(run_info)
        source_root = run_info_path
    if not candidates:
        summary_paths = summary_waveform_paths(config)
        if summary_paths:
            warn(
                "incident power skipped; frozen JSON contains summary-only waveform "
                f"at {', '.join(summary_paths)} without tabulated time/value arrays"
            )
        else:
            warn(
                "incident power skipped; no tabulated laser waveform in frozen JSON "
                "or run_info.json"
            )
        return None, []

    total_tw = np.zeros_like(times_s, dtype=float)
    sources = []
    for path, table_time, table_power, units in candidates:
        values = np.interp(
            times_s,
            table_time,
            table_power,
            left=table_power[0],
            right=table_power[-1],
        )
        lowered_units = units.lower().replace(" ", "")
        if "erg/s" in lowered_units or "ergs-1" in lowered_units:
            total_tw += values * ERG_PER_S_TO_TW
        else:
            # Laser beam power callables and their frozen tables use W.
            total_tw += values * W_TO_TW
        sources.append(f"{source_root}:{path}")
    print(f"INFO: incident power source: {', '.join(sources)}")
    return total_tw, sources


def integrated_power_tw(field, volume):
    values = np.asarray(field, dtype=float).reshape(-1)
    volume = np.asarray(volume, dtype=float).reshape(-1)
    if values.size != volume.size or values.size == 0:
        return np.nan
    valid = np.isfinite(values) & np.isfinite(volume)
    if not np.any(valid):
        return np.nan
    return float(np.sum(values[valid] * volume[valid]) * ERG_PER_S_TO_TW)


def channel_labels(config, n_channels):
    channels = frozen_channels(config)
    labels = []
    for index in range(n_channels):
        if index < len(channels) and isinstance(channels[index], dict):
            labels.append(str(channels[index].get("mechanism", f"ch{index}")).upper())
        else:
            labels.append(f"ch{index}")
    return labels


def plot_fig2(
    snapshots,
    config,
    frozen_path,
    run_info,
    run_info_path,
    out_dir,
    dpi,
):
    name = "fig2_energetics.png"
    rows = []
    used = []
    warned_missing = set()
    n_channels = None

    for snapshot in snapshots:
        row = {
            "time_s": snapshot["time_s"],
            "IB": np.nan,
            "CBET": np.nan,
            "hot": np.nan,
            "eta": None,
            "g": None,
        }
        try:
            with h5py.File(snapshot["path"], "r") as handle:
                volume = get(handle, HYDRO_VOL)
                if volume is None:
                    warned_missing.add(HYDRO_VOL)
                else:
                    power_fields = {
                        "IB": (LASER_DEPOSITED_POWER, get(handle, LASER_DEPOSITED_POWER)),
                        "CBET": (HYDRO_CBET_GROSS, get(handle, HYDRO_CBET_GROSS)),
                        "hot": (HYDRO_HOT_E_Q, get(handle, HYDRO_HOT_E_Q)),
                    }
                    for key, (path, value) in power_fields.items():
                        if value is None:
                            warned_missing.add(path)
                            continue
                        row[key] = integrated_power_tw(value, volume)
                        if np.isfinite(row[key]):
                            used.extend((path, HYDRO_VOL))

                eta_value = get(handle, HYDRO_ETA)
                g_value = get(handle, HYDRO_G)
                if eta_value is None:
                    warned_missing.add(HYDRO_ETA)
                if g_value is None:
                    warned_missing.add(HYDRO_G)
                if eta_value is not None and g_value is not None:
                    eta = np.asarray(eta_value, dtype=float).reshape(-1)
                    g = np.asarray(g_value, dtype=float).reshape(-1)
                    if eta.size == g.size and eta.size > 0:
                        if n_channels is None:
                            n_channels = eta.size
                        if eta.size == n_channels:
                            row["eta"] = eta
                            row["g"] = g
                            used.extend((HYDRO_ETA, HYDRO_G))
                        else:
                            warn(
                                f"{name}: omitted eta/g from {snapshot['path'].name}; "
                                "channel count changed"
                            )
                    else:
                        warn(
                            f"{name}: omitted eta/g from {snapshot['path'].name}; "
                            "channel dimensions disagree"
                        )
        except OSError as exc:
            warn(f"{name}: omitted unreadable snapshot {snapshot['path']}: {exc}")
            continue
        rows.append(row)
        used.append(snapshot["time_source"])

    for missing_path in sorted(warned_missing):
        warn(f"{name}: omitted unavailable samples for missing dataset {missing_path}")
    if not rows:
        warn_skipped(name, [str(snapshots[0]["path"].parent / "*.h5")])
        return 0

    times_s = np.asarray([row["time_s"] for row in rows])
    times_ns = times_s * 1.0e9
    incident_tw, incident_sources = incident_power_series(
        config, run_info, times_s, frozen_path, run_info_path
    )
    used.extend(incident_sources)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.5, 8.2),
        sharex=True,
        constrained_layout=True,
    )
    power_axis, eta_axis = axes
    curve_count = 0
    if incident_tw is not None and np.any(np.isfinite(incident_tw)):
        power_axis.plot(
            times_ns,
            incident_tw,
            color="black",
            linewidth=2.5,
            label="入射",
        )
        curve_count += 1
    power_specs = [
        ("IB", "吸収 (IB)", "#0072B2"),
        ("CBET", "CBET 交換", "#E69F00"),
        ("hot", "hot-e 変換", "#D55E00"),
    ]
    for key, label, color in power_specs:
        values = np.asarray([row[key] for row in rows], dtype=float)
        if np.any(np.isfinite(values)):
            power_axis.plot(
                times_ns,
                values,
                color=color,
                linewidth=1.7,
                label=label,
            )
            curve_count += 1
            if key == "hot":
                power_axis.plot(
                    times_ns,
                    100.0 * values,
                    color="#D62728",
                    linestyle="--",
                    linewidth=1.7,
                    label="hot-e 変換 ×100",
                )
                curve_count += 1
    if not curve_count:
        plt.close(fig)
        warn_skipped(
            name,
            [LASER_DEPOSITED_POWER, HYDRO_CBET_GROSS, HYDRO_HOT_E_Q],
        )
        return 0
    power_axis.set_ylabel("電力 [TW]")
    power_axis.set_title("(a) 電力収支 — x3 drive demo", loc="left")
    power_axis.grid(True, alpha=0.25)
    power_axis.legend(loc="best")
    power_axis.set_ylim(bottom=0.0)

    valid_channel_rows = [row for row in rows if row["eta"] is not None]
    if valid_channel_rows and n_channels:
        plotted_channels = min(n_channels, 2)
        if n_channels > plotted_channels:
            warn(
                f"{name}: clarity limit omitted channels {plotted_channels}.."
                f"{n_channels - 1}; maximum four curves per panel"
            )
        eta_times_ns = (
            np.asarray([row["time_s"] for row in valid_channel_rows]) * 1.0e9
        )
        eta_series = np.asarray([row["eta"] for row in valid_channel_rows])
        g_series = np.asarray([row["g"] for row in valid_channel_rows])
        labels = channel_labels(config, n_channels)
        colors = ("#0072B2", "#D55E00")
        g_axis = eta_axis.twinx()
        burst_labeled = False
        combined_lines = []
        for channel in range(plotted_channels):
            color = colors[channel]
            eta_line = eta_axis.plot(
                eta_times_ns,
                100.0 * eta_series[:, channel],
                color=color,
                linewidth=1.8,
                label=f"{labels[channel]} η",
            )[0]
            g_line = g_axis.plot(
                eta_times_ns,
                g_series[:, channel],
                color=color,
                linestyle="--",
                linewidth=1.1,
                alpha=0.68,
                label=f"{labels[channel]} g",
            )[0]
            combined_lines.extend((eta_line, g_line))

            above = np.isfinite(g_series[:, channel]) & (
                g_series[:, channel] >= 1.0
            )
            previous = np.concatenate(([False], above[:-1]))
            onsets = np.flatnonzero(above & ~previous)
            if onsets.size:
                eta_axis.scatter(
                    eta_times_ns[onsets],
                    100.0 * eta_series[onsets, channel],
                    marker="*",
                    s=55,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.4,
                    zorder=5,
                )
                if not burst_labeled:
                    first = int(onsets[0])
                    eta_axis.annotate(
                        "バースト",
                        xy=(
                            eta_times_ns[first],
                            100.0 * eta_series[first, channel],
                        ),
                        xytext=(7, 12),
                        textcoords="offset points",
                        fontsize=9,
                        arrowprops={"arrowstyle": "-", "color": color},
                    )
                    burst_labeled = True
        g_axis.axhline(1.0, color="black", linestyle=":", linewidth=1.1)
        g_axis.text(
            0.99,
            1.0,
            "閾値",
            transform=g_axis.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=9,
        )
        eta_axis.set_ylabel("η [%]")
        g_axis.set_ylabel("g")
        eta_axis.set_ylim(bottom=0.0)
        eta_axis.grid(True, alpha=0.25)
        eta_axis.legend(
            combined_lines,
            [line.get_label() for line in combined_lines],
            loc="best",
            ncol=2,
            fontsize=8,
        )
    else:
        eta_axis.text(
            0.5,
            0.5,
            "η/g データなし",
            transform=eta_axis.transAxes,
            ha="center",
            va="center",
        )
        warn(f"{name}: eta/g panel has no complete channel samples")
    eta_axis.set_title("(b) チャネル別 η と閾値比 g", loc="left")
    eta_axis.set_xlabel("時間 t [ns]")

    path = out_dir / name
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    written(path, used)
    return 1


def is_icosahedral_port_table(axes):
    if axes.shape != (12, 3):
        return False
    dots = axes @ axes.T
    for index in range(axes.shape[0]):
        other = np.delete(dots[index], index)
        if not np.any(np.isclose(other, -1.0, atol=1.0e-7)):
            return False
        non_antipodal = other[~np.isclose(other, -1.0, atol=1.0e-7)]
        if not np.all(
            np.isclose(np.abs(non_antipodal), 1.0 / np.sqrt(5.0), atol=1.0e-7)
        ):
            return False
    return True


def port_caption(ports):
    caption = f"{len(ports['ids'])} 港"
    if is_icosahedral_port_table(ports["axes"]):
        caption += " icosahedral"
    properties = []
    if np.allclose(ports["weights"], ports["weights"][0]):
        properties.append("等重み")
    if np.allclose(ports["delta_lambda"], ports["delta_lambda"][0]):
        properties.append("同波長")
    if properties:
        caption += "・" + "・".join(properties)
    return caption


def plot_port_hemispheres(axis, ports):
    centers = (-1.25, 1.25)
    labels = ("z ≥ 0", "z < 0")
    for center, label in zip(centers, labels):
        axis.add_patch(
            Circle(
                (center, 0.0),
                1.0,
                facecolor="#F5F5F5",
                edgecolor="black",
                linewidth=1.0,
            )
        )
        axis.text(center, 1.10, label, ha="center", va="bottom", fontsize=9)

    maximum_weight = max(float(np.max(ports["weights"])), 1.0e-30)
    for index, (direction, weight, port_id) in enumerate(
        zip(ports["axes"], ports["weights"], ports["ids"])
    ):
        hemisphere = 0 if direction[2] >= 0.0 else 1
        center = centers[hemisphere]
        # Lambert azimuthal equal-area projection about each pole:
        # k=sqrt(2/(1+|z|)), (X,Y)=k*(x,y), normalized by sqrt(2).
        denominator = np.sqrt(max(1.0 + abs(float(direction[2])), 1.0e-30))
        x_plot = center + float(direction[0]) / denominator
        y_plot = float(direction[1]) / denominator
        size = 75.0 + 210.0 * float(weight) / maximum_weight
        axis.scatter(
            x_plot,
            y_plot,
            s=size,
            color="#56B4E9",
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
        )
        axis.text(
            x_plot,
            y_plot,
            str(port_id),
            ha="center",
            va="center",
            fontsize=7,
            zorder=4,
        )
    axis.text(
        0.5,
        -0.08,
        port_caption(ports),
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    axis.set_xlim(-2.4, 2.4)
    axis.set_ylim(-1.25, 1.25)
    axis.set_aspect("equal")
    axis.axis("off")


def mu_quadrature_weights(mu):
    mu = np.asarray(mu, dtype=float)
    count = mu.size
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(count)
    if np.allclose(mu, gauss_nodes, rtol=1.0e-12, atol=1.0e-14):
        return gauss_weights
    edges = np.empty(count + 1, dtype=float)
    edges[0] = -1.0
    edges[-1] = 1.0
    edges[1:-1] = 0.5 * (mu[:-1] + mu[1:])
    return np.maximum(np.diff(edges), 0.0)


def periodic_phi_weights(phi):
    phi = np.asarray(phi, dtype=float)
    if phi.size == 1:
        return np.asarray([2.0 * PI])
    previous = np.concatenate(([phi[-1] - 2.0 * PI], phi[:-1]))
    following = np.concatenate((phi[1:], [phi[0] + 2.0 * PI]))
    return 0.5 * (following - previous)


def port_mollweide_coordinates(ports):
    axes = ports["axes"]
    longitude = np.arctan2(axes[:, 1], axes[:, 0])
    latitude = np.arcsin(np.clip(axes[:, 2], -1.0, 1.0))
    return longitude, latitude


def add_port_markers(axis, ports):
    longitude, latitude = port_mollweide_coordinates(ports)
    axis.scatter(
        longitude,
        latitude,
        marker="x",
        s=38,
        linewidths=2.4,
        color="black",
        zorder=5,
    )
    axis.scatter(
        longitude,
        latitude,
        marker="x",
        s=38,
        linewidths=1.1,
        color="white",
        zorder=6,
    )


def plot_fig3(snapshot, ports, frozen_path, out_dir, dpi):
    name = "fig3_multibeam.png"
    if ports is None:
        warn_skipped(name, ["frozen:laser.port_configuration.ports"])
        return 0
    required = {
        "mu": SKY_MU,
        "phi": SKY_PHI,
        "I_tot": SKY_I_TOT,
        "I_cw": SKY_I_CW,
    }
    try:
        with h5py.File(snapshot["path"], "r") as handle:
            values = {}
            missing = []
            for key, path in required.items():
                value = get(handle, path)
                if value is None:
                    missing.append(path)
                else:
                    values[key] = value
    except OSError as exc:
        warn(f"skipping {name}; could not read {snapshot['path']}: {exc}")
        return 0
    if missing:
        warn_skipped(name, missing)
        return 0

    mu = np.asarray(values["mu"], dtype=float).reshape(-1)
    phi = np.mod(np.asarray(values["phi"], dtype=float).reshape(-1), 2.0 * PI)
    i_tot = np.asarray(values["I_tot"], dtype=float)
    i_cw = np.asarray(values["I_cw"], dtype=float)
    expected_shape = (mu.size, phi.size)
    if (
        mu.size < 2
        or phi.size < 2
        or i_tot.shape != expected_shape
        or i_cw.shape != expected_shape
        or not np.all(np.isfinite(mu))
        or not np.all(np.isfinite(phi))
    ):
        warn_skipped(name, ["compatible grid_mu/grid_phi/sky-map dimensions"])
        return 0

    mu_order = np.argsort(mu)
    longitude = np.mod(phi + PI, 2.0 * PI) - PI
    phi_order = np.argsort(longitude)
    mu = mu[mu_order]
    longitude = longitude[phi_order]
    i_tot = i_tot[np.ix_(mu_order, phi_order)]
    i_cw = i_cw[np.ix_(mu_order, phi_order)]
    latitude = np.arcsin(np.clip(mu, -1.0, 1.0))
    longitude_grid, latitude_grid = np.meshgrid(longitude, latitude)

    mu_weights = mu_quadrature_weights(mu)
    phi_weights = periodic_phi_weights(longitude)
    weights = mu_weights[:, np.newaxis] * phi_weights[np.newaxis, :]
    valid_tot = np.isfinite(i_tot)
    weight_sum = float(np.sum(weights[valid_tot]))
    if weight_sum > 0.0:
        mean_intensity = float(np.sum(weights[valid_tot] * i_tot[valid_tot]) / weight_sum)
        variance = float(
            np.sum(
                weights[valid_tot]
                * (i_tot[valid_tot] - mean_intensity) ** 2
            )
            / weight_sum
        )
        sigma_rms = (
            100.0 * np.sqrt(max(variance, 0.0)) / mean_intensity
            if mean_intensity > 0.0
            else np.nan
        )
    else:
        sigma_rms = np.nan

    ratio = np.divide(
        i_cw,
        i_tot,
        out=np.full_like(i_cw, np.nan),
        where=np.isfinite(i_tot) & np.isfinite(i_cw) & (i_tot > 0.0),
    )
    finite_tot = i_tot[np.isfinite(i_tot)]
    tot_vmax = float(np.max(finite_tot)) if finite_tot.size else 1.0
    finite_ratio = ratio[np.isfinite(ratio)]
    ratio_vmax = max(
        1.2,
        float(np.percentile(finite_ratio, 99.0)) if finite_ratio.size else 1.2,
    )

    fig = plt.figure(figsize=(17.0, 5.4), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(0.85, 1.15, 1.15))
    port_axis = fig.add_subplot(grid[0, 0])
    total_axis = fig.add_subplot(grid[0, 1], projection="mollweide")
    ratio_axis = fig.add_subplot(grid[0, 2], projection="mollweide")

    plot_port_hemispheres(port_axis, ports)
    port_axis.set_title("(a) ポート配置", loc="left")

    total_mesh = total_axis.pcolormesh(
        longitude_grid,
        latitude_grid,
        i_tot,
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=max(tot_vmax, 1.0e-30),
        rasterized=True,
    )
    add_port_markers(total_axis, ports)
    total_axis.grid(True, alpha=0.25)
    total_axis.tick_params(labelbottom=False, labelleft=False)
    total_axis.set_title("(b) 全照射強度 I_tot")
    total_axis.text(
        0.03,
        0.04,
        f"σ_rms = {sigma_rms:.2f} %",
        transform=total_axis.transAxes,
        ha="left",
        va="bottom",
        color="white",
        bbox={"facecolor": "black", "alpha": 0.48, "edgecolor": "none"},
    )
    fig.colorbar(
        total_mesh,
        ax=total_axis,
        orientation="horizontal",
        pad=0.08,
        label="I_tot [W/cm²]",
    )

    ratio_mesh = ratio_axis.pcolormesh(
        longitude_grid,
        latitude_grid,
        np.ma.masked_invalid(ratio),
        shading="auto",
        cmap="magma",
        vmin=0.9,
        vmax=ratio_vmax,
        rasterized=True,
    )
    add_port_markers(ratio_axis, ports)
    ratio_axis.grid(True, alpha=0.25)
    ratio_axis.tick_params(labelbottom=False, labelleft=False)
    ratio_axis.set_title("(c) I_cw/I_tot")
    if finite_ratio.size:
        maximum_index = np.unravel_index(np.nanargmax(ratio), ratio.shape)
        ratio_axis.annotate(
            "common-wave 増強域",
            xy=(
                longitude_grid[maximum_index],
                latitude_grid[maximum_index],
            ),
            xytext=(0.04, 0.08),
            textcoords="axes fraction",
            ha="left",
            va="bottom",
            fontsize=9,
            color="white",
            arrowprops={"arrowstyle": "-", "color": "white", "linewidth": 0.9},
            bbox={"facecolor": "black", "alpha": 0.40, "edgecolor": "none"},
        )
    fig.colorbar(
        ratio_mesh,
        ax=ratio_axis,
        orientation="horizontal",
        pad=0.08,
        label="I_cw/I_tot",
    )
    fig.suptitle(f"多ビーム照射  t = {snapshot['time_s'] * 1.0e9:.3f} ns")

    path = out_dir / name
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    used = list(required.values()) + [snapshot["time_source"]]
    if frozen_path is not None:
        used.append(f"{frozen_path}:laser.port_configuration.ports")
    written(path, used)
    return 1


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create literature-style multibeam summary figures."
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
        "--time",
        type=float,
        default=None,
        metavar="T_NS",
        help="snapshot time in ns (default: nearest 0.75*t_last)",
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
    if args.time is not None and not np.isfinite(args.time):
        warn("--time must be finite")
        return 2
    if args.dpi <= 0:
        warn("--dpi must be positive")
        return 2

    if args.out is None:
        out_dir = (
            pathlib.Path("~/work/projects/TENRYU/outputs/plots").expanduser()
            / f"{run_dir.resolve().name}_summary"
        )
    else:
        out_dir = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshots = discover_snapshots(run_dir)
    if not snapshots:
        warn_skipped("all figures", [str(run_dir / "results" / "*.h5")])
        return 2
    snapshot = selected_snapshot(snapshots, args.time)
    print(
        f"INFO: selected snapshot {snapshot['path'].name} at "
        f"{snapshot['time_s'] * 1.0e9:.6f} ns"
    )

    config, frozen_path = load_frozen_config(run_dir)
    run_info, run_info_path = load_run_info(run_dir)
    ports = normalized_ports(config)

    figure_count = 0
    figure_count += plot_fig1(
        snapshot, config, ports, frozen_path, out_dir, args.dpi
    )
    figure_count += plot_fig2(
        snapshots,
        config,
        frozen_path,
        run_info,
        run_info_path,
        out_dir,
        args.dpi,
    )
    figure_count += plot_fig3(
        snapshot, ports, frozen_path, out_dir, args.dpi
    )
    if figure_count == 0:
        warn("no figures were produced")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
