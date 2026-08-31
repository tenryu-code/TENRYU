"""Deterministic one-dimensional ablation-zoning report generation."""

import bisect
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Optional

from tools.assist.digest import _values


_ONLY_1D_REASON = "only 1D outputs are supported in this milestone"
_LINT_CAVEATS = [
    "snapshot cadence limits L_c minimum",
    "kappa/margins provisional until probe calibration",
]


def _read_json_dict(path: Path, notes: list, label: str) -> Optional[dict]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        notes.append("{0} could not be read: {1}".format(label, error))
        return None
    if not isinstance(value, dict):
        notes.append("{0} is not a JSON object".format(label))
        return None
    return value


def _nested_value(values: dict, section: str, key: str):
    nested = values.get(section)
    if not isinstance(nested, dict):
        return None
    return nested.get(key)


def _base_report(root: Path, kappa: float, ablate_margin: float, notes: list) -> dict:
    return {
        "schema": "tenryu.assist.zoning.v0",
        "output_dir": str(root),
        "applicable": False,
        "reason": None,
        "geometry": None,
        "n_snapshots_used": 0,
        "kappa": float(kappa),
        "ablate_margin": float(ablate_margin),
        "critical": None,
        "ablated": None,
        "lint_a": None,
        "pressure_track": None,
        "notes": notes,
    }


def _float_list(dataset) -> list:
    if len(tuple(dataset.shape)) != 1:
        raise ValueError("dataset is not one-dimensional")
    return [float(value) for value in dataset[...]]


def _decoded_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _snapshot_paths(root: Path) -> list:
    return [
        path
        for path in sorted((root / "results").glob("*.h5"))
        if "_history" not in path.name and "_ckpt_" not in path.name
    ]


def _read_snapshot(path: Path, h5py_module, notes: list) -> Optional[dict]:
    try:
        with h5py_module.File(str(path), "r") as handle:
            if "t" not in handle.attrs or "geometry" not in handle.attrs:
                raise ValueError("required root attribute missing")
            if "mesh/x_r" not in handle or "hydro/rho" not in handle:
                raise ValueError("required dataset missing")

            time_value = float(handle.attrs["t"])
            geometry = _decoded_string(handle.attrs["geometry"])
            nodes = _float_list(handle["mesh/x_r"])
            rho = _float_list(handle["hydro/rho"])
            if len(nodes) != len(rho) + 1 or not rho:
                raise ValueError("mesh/x_r and hydro/rho lengths are inconsistent")

            centers = [
                0.5 * (nodes[index] + nodes[index + 1])
                for index in range(len(rho))
            ]
            widths = [
                nodes[index + 1] - nodes[index]
                for index in range(len(rho))
            ]
            snapshot = {
                "path": path,
                "t": time_value,
                "geometry": geometry,
                "rho": rho,
                "centers": centers,
                "widths": widths,
                "Pe": None,
                "Pi": None,
            }
            for name in ("Pe", "Pi"):
                dataset_name = "hydro/{0}".format(name)
                if dataset_name not in handle:
                    continue
                values = _float_list(handle[dataset_name])
                if len(values) != len(rho):
                    notes.append(
                        "snapshot {0}: {1} length mismatch — pressure track skipped".format(
                            path.name, dataset_name
                        )
                    )
                    continue
                snapshot[name] = values
            return snapshot
    except (OSError, ValueError, TypeError, KeyError) as error:
        notes.append(
            "snapshot {0} missing or invalid required pieces — skipped: {1}".format(
                path.name, error
            )
        )
        return None


def _read_critical_history(path: Path, h5py_module, notes: list) -> list:
    try:
        with h5py_module.File(str(path), "r") as handle:
            if "t" not in handle or "laser/critical_surface_r" not in handle:
                return []
            times = _float_list(handle["t"])
            radii = _float_list(handle["laser/critical_surface_r"])
    except (OSError, ValueError, TypeError, KeyError) as error:
        notes.append("history could not be read: {0}".format(error))
        return []

    if len(times) != len(radii):
        notes.append("history t and laser/critical_surface_r lengths differ")
    valid = []
    for time_value, radius in zip(times, radii):
        if math.isfinite(time_value) and math.isfinite(radius) and radius > 0.0:
            valid.append((time_value, radius))
    valid.sort(key=lambda pair: pair[0])
    return valid


def _interpolate(xs: list, ys: list, value: float) -> float:
    if len(xs) == 1:
        return ys[0]
    if value <= xs[0]:
        return ys[0]
    if value >= xs[-1]:
        return ys[-1]
    upper = bisect.bisect_right(xs, value)
    lower = upper - 1
    if xs[upper] == xs[lower]:
        return ys[upper]
    fraction = (value - xs[lower]) / (xs[upper] - xs[lower])
    return ys[lower] + fraction * (ys[upper] - ys[lower])


def _scale_length(snapshot: dict, radius: float, notes: list) -> Optional[float]:
    centers = snapshot["centers"]
    rho = snapshot["rho"]
    bracket = bisect.bisect_right(centers, radius) - 1
    bracket = max(0, min(bracket, len(centers) - 1))
    start = max(0, bracket - 2)
    stop = min(len(centers) - 1, bracket + 3)
    points = [
        (centers[index], math.log(rho[index]))
        for index in range(start, stop + 1)
        if rho[index] > 0.0
    ]
    if len(points) < 3:
        notes.append(
            "snapshot {0}: fewer than 3 positive-density cells in critical window".format(
                snapshot["path"].name
            )
        )
        return None

    x_mean = sum(point[0] for point in points) / float(len(points))
    y_mean = sum(point[1] for point in points) / float(len(points))
    denominator = sum((point[0] - x_mean) ** 2 for point in points)
    if denominator == 0.0:
        notes.append(
            "snapshot {0}: critical density-fit slope is zero".format(
                snapshot["path"].name
            )
        )
        return None
    slope = sum(
        (point[0] - x_mean) * (point[1] - y_mean) for point in points
    ) / denominator
    if slope == 0.0:
        notes.append(
            "snapshot {0}: critical density-fit slope is zero".format(
                snapshot["path"].name
            )
        )
        return None
    return 1.0 / abs(slope)


def _ablated_cell_indices(
    snapshots: list, rho_c: float, ablate_margin: float, notes: list
) -> list:
    total_cells = len(snapshots[0]["rho"])
    matching_snapshots = []
    for snapshot in snapshots:
        if len(snapshot["rho"]) != total_cells:
            notes.append(
                "snapshot {0}: cell count differs from first snapshot — skipped for ablated set".format(
                    snapshot["path"].name
                )
            )
            continue
        matching_snapshots.append(snapshot)

    rho_min = [float("inf")] * total_cells
    for snapshot in matching_snapshots:
        for index, value in enumerate(snapshot["rho"]):
            rho_min[index] = min(rho_min[index], value)
    return [
        index
        for index, value in enumerate(rho_min)
        if value < float(ablate_margin) * rho_c
    ]


def compute_band_promotion(
    mass,
    x_r,
    ablated_indices,
    threshold_areal_g_cm2,
    margin_cells=3,
):
    """Compile an observed ablated set into a hard spherical-measure band.

    mass: per-cell initial masses [g] (Lagrangian invariants of the run).
    x_r: initial node radii [cm], len(mass)+1.
    ablated_indices: iterable of cell indices observed to ablate.
    threshold_areal_g_cm2: kappa*rho_c*L_c_min from the zoning report.
    Returns the promotion dict or raises ValueError on empty inputs.
    """
    masses = [float(value) for value in mass]
    nodes = [float(value) for value in x_r]
    indices = sorted(set(int(index) for index in ablated_indices))
    if not masses or not nodes or not indices:
        raise ValueError("mass, x_r, and ablated_indices must be non-empty")

    margin = int(margin_cells)
    first_ablated = min(indices)
    band_start_cell = max(0, first_ablated - margin)
    total = sum(masses)
    if band_start_cell == 0:
        measure_frac_begin = 0.0
    else:
        measure_frac_begin = sum(masses[:band_start_cell]) / total
    r0 = nodes[band_start_cell]
    threshold = float(threshold_areal_g_cm2)
    cell_measure_max = threshold * 4.0 * math.pi * r0 * r0

    return {
        "band": {
            "measure_frac_begin": measure_frac_begin,
            "measure_frac_end": 1.0,
            "cell_measure_max": cell_measure_max,
        },
        "evidence": {
            "first_ablated_cell": first_ablated,
            "band_start_cell": band_start_cell,
            "margin_cells": margin,
            "n_ablated": len(indices),
            "threshold_areal_g_cm2": threshold,
            "r0_cm": r0,
            "initial_mass_frac_at_band_start": measure_frac_begin,
        },
        "conservatism_note": (
            "cell_measure_max is exact at the band's inner radius r0 and tighter "
            "than the areal threshold by (r/r0)^2 at larger radii; a run whose "
            "ablation front reaches deeper than the probe's needs a re-probe"
        ),
    }


def build_zoning_report(
    output_dir: str, kappa: float = 1.0, ablate_margin: float = 1.2
) -> dict:
    """Build a JSON-safe ablation-zoning report from an output directory."""
    root = Path(output_dir).resolve()
    notes = []
    report = _base_report(root, kappa, ablate_margin, notes)

    run_info_path = root / "run_info.json"
    if run_info_path.is_file():
        _read_json_dict(run_info_path, notes, "run_info.json")
    else:
        notes.append("run_info.json not found")

    frozen_paths = sorted((root / "config").glob("*_frozen.json"))
    frozen_values = None
    if not frozen_paths:
        notes.append("frozen config not found")
    else:
        if len(frozen_paths) > 1:
            notes.append(
                "multiple frozen configs found; using lexicographically first: {0}".format(
                    frozen_paths[0]
                )
            )
        frozen_values = _read_json_dict(frozen_paths[0], notes, "frozen config")

    dimension = (
        _nested_value(frozen_values, "main", "dimension")
        if frozen_values is not None
        else None
    )
    is_1d = dimension == 1 or (
        isinstance(dimension, str) and dimension.upper().startswith("1D")
    )
    if dimension is not None and not is_1d:
        report["reason"] = _ONLY_1D_REASON
        return report

    mesh_motion = (
        _nested_value(frozen_values, "mesh", "motion")
        if frozen_values is not None
        else None
    )
    if mesh_motion is not None and mesh_motion != "lagrangian":
        notes.append(
            "mesh motion is not lagrangian — cell identity across snapshots is approximate"
        )

    try:
        import h5py
    except ImportError:
        report["reason"] = "h5py unavailable"
        return report

    snapshot_paths = _snapshot_paths(root)
    snapshots = []
    for path in snapshot_paths:
        snapshot = _read_snapshot(path, h5py, notes)
        if snapshot is not None:
            snapshots.append(snapshot)

    report["n_snapshots_used"] = len(snapshots)
    if snapshots:
        report["geometry"] = snapshots[0]["geometry"]
    else:
        report["reason"] = "no usable snapshots"
        return report

    history_paths = sorted((root / "results").glob("*_history.h5"))
    if not history_paths:
        report["reason"] = "no laser critical-surface record"
        return report
    if len(history_paths) > 1:
        notes.append(
            "multiple history files found; using lexicographically first: {0}".format(
                history_paths[0]
            )
        )
    critical_samples = _read_critical_history(history_paths[0], h5py, notes)
    if not critical_samples:
        report["reason"] = "no laser critical-surface record"
        return report

    history_times = [sample[0] for sample in critical_samples]
    history_radii = [sample[1] for sample in critical_samples]
    critical_track = []
    for snapshot in snapshots:
        critical_radius = _interpolate(
            history_times, history_radii, snapshot["t"]
        )
        centers = snapshot["centers"]
        if critical_radius < centers[0] or critical_radius > centers[-1]:
            notes.append(
                "snapshot {0}: critical radius outside cell-center range — skipped for critical fit".format(
                    snapshot["path"].name
                )
            )
            continue
        rho_at_radius = _interpolate(centers, snapshot["rho"], critical_radius)
        scale_length = _scale_length(snapshot, critical_radius, notes)
        critical_track.append(
            {
                "t_s": snapshot["t"],
                "r_c_cm": critical_radius,
                "rho_at_rc_gcc": rho_at_radius,
                "L_c_cm": scale_length,
            }
        )

    rho_values = [entry["rho_at_rc_gcc"] for entry in critical_track]
    scale_lengths = [
        entry["L_c_cm"]
        for entry in critical_track
        if entry["L_c_cm"] is not None
    ]
    rho_c = statistics.median(rho_values) if rho_values else None
    scale_length_min = min(scale_lengths) if scale_lengths else None
    report["critical"] = {
        "rho_c_gcc": rho_c,
        "L_c_min_cm": scale_length_min,
        "track": critical_track,
    }
    report["applicable"] = True

    if rho_c is None:
        return report

    total_cells = len(snapshots[0]["rho"])
    ablated_indices = _ablated_cell_indices(
        snapshots, rho_c, ablate_margin, notes
    )
    report["ablated"] = {
        "n_cells": len(ablated_indices),
        "n_total_cells": total_cells,
    }

    first_snapshot = snapshots[0]
    areal_masses = [
        density * width
        for density, width in zip(
            first_snapshot["rho"], first_snapshot["widths"]
        )
    ]
    if ablated_indices:
        maximum_index = max(ablated_indices, key=lambda index: areal_masses[index])
        maximum_areal_mass = areal_masses[maximum_index]
    else:
        maximum_index = None
        maximum_areal_mass = None

    if scale_length_min is None:
        threshold = None
        violations = []
        lint_ok = None
        notes.append("scale length unavailable")
    else:
        threshold = float(kappa) * rho_c * scale_length_min
        violations = [
            index
            for index in ablated_indices
            if areal_masses[index] > threshold
        ]
        lint_ok = len(violations) == 0
    report["lint_a"] = {
        "threshold_g_cm2": threshold,
        "max_areal_mass_g_cm2": maximum_areal_mass,
        "at_index": maximum_index,
        "n_violations": len(violations),
        "violating_indices_head": violations[:10],
        "ok": lint_ok,
        "caveats": list(_LINT_CAVEATS),
    }

    pressure_entries = []
    for snapshot in snapshots:
        if snapshot["Pe"] is None or snapshot["Pi"] is None:
            continue
        overdense_pressures = [
            snapshot["Pe"][index] + snapshot["Pi"][index]
            for index, density in enumerate(snapshot["rho"])
            if density > rho_c
        ]
        if not overdense_pressures:
            notes.append(
                "snapshot {0}: no overdense cells for pressure track".format(
                    snapshot["path"].name
                )
            )
            continue
        pressure_entries.append(
            {
                "t_s": snapshot["t"],
                "p_peak_overdense": max(overdense_pressures),
            }
        )
    report["pressure_track"] = {
        "track": pressure_entries,
        "uncalibrated_indicator": True,
    }
    return report


def promote_zoning(output_dir, kappa=None, margin_cells=3):
    """Compile a zoning-report observation into a hard mesh band."""
    root = Path(output_dir).resolve()
    if not root.is_dir():
        return (
            {
                "applicable": False,
                "reason": (
                    "output directory does not exist or is not a directory: "
                    "{0}".format(root)
                ),
            },
            2,
        )

    effective_kappa = 1.0 if kappa is None else kappa
    report = build_zoning_report(str(root), kappa=effective_kappa)
    if not report["applicable"]:
        return {
            "applicable": False,
            "reason": report["reason"],
        }, 0
    if report["geometry"] != "1D_SPH":
        return {
            "applicable": False,
            "reason": "promotion currently emits spherical-measure bands only",
        }, 0

    critical = report["critical"]
    lint_a = report["lint_a"]
    if critical is None or critical["rho_c_gcc"] is None or lint_a is None:
        return {
            "applicable": False,
            "reason": "zoning report did not produce ablation inputs",
        }, 2
    if critical["L_c_min_cm"] is None or lint_a["threshold_g_cm2"] is None:
        return {
            "applicable": False,
            "reason": "zoning report did not produce an areal-mass threshold",
        }, 2

    try:
        import h5py
    except ImportError:
        return {
            "applicable": False,
            "reason": "h5py unavailable",
        }, 2

    notes = []
    snapshots = []
    for path in _snapshot_paths(root):
        snapshot = _read_snapshot(path, h5py, notes)
        if snapshot is not None:
            snapshots.append(snapshot)
    ablated_indices = _ablated_cell_indices(
        snapshots, critical["rho_c_gcc"], 1.2, notes
    )
    if not ablated_indices:
        return {
            "applicable": False,
            "reason": "no ablated cells observed",
        }, 0

    initial_path = snapshots[0]["path"]
    try:
        with h5py.File(str(initial_path), "r") as handle:
            if "hydro/mass" not in handle or "mesh/x_r" not in handle:
                raise ValueError("required initial-state dataset missing")
            mass = _values(handle["hydro/mass"], "hydro/mass", notes, None)
            x_r = _values(handle["mesh/x_r"], "mesh/x_r", notes, None)
            if mass is None or x_r is None:
                raise ValueError("initial-state dataset is not a readable series")
    except (OSError, ValueError, TypeError, KeyError) as error:
        return {
            "applicable": False,
            "reason": "initial snapshot could not be read: {0}".format(error),
        }, 2

    try:
        promotion = compute_band_promotion(
            mass,
            x_r,
            ablated_indices,
            lint_a["threshold_g_cm2"],
            margin_cells=margin_cells,
        )
    except (IndexError, TypeError, ValueError, ZeroDivisionError) as error:
        return {
            "applicable": False,
            "reason": "promotion inputs are invalid: {0}".format(error),
        }, 2

    band = promotion["band"]
    suggested_patch = (
        '{"measure_frac_begin": '
        + repr(band["measure_frac_begin"])
        + ', "measure_frac_end": '
        + repr(band["measure_frac_end"])
        + ', "cell_measure_max": '
        + repr(band["cell_measure_max"])
        + "}"
    )
    return {
        "schema": "tenryu.assist.zoning_promote.v0",
        "output_dir": str(root),
        "applicable": True,
        "kappa": report["kappa"],
        "rho_c_gcc": critical["rho_c_gcc"],
        "L_c_min_cm": critical["L_c_min_cm"],
        "n_snapshots_used": report["n_snapshots_used"],
        "band": band,
        "evidence": promotion["evidence"],
        "conservatism_note": promotion["conservatism_note"],
        "suggested_patch": suggested_patch,
    }, 0


def main_zoning_report(args) -> int:
    """CLI entry point for the zoning-report verb."""
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(output_dir):
        print(
            "assist zoning-report: output directory does not exist or is not a directory: "
            "{0}".format(output_dir),
            file=sys.stderr,
        )
        return 2

    report = build_zoning_report(
        output_dir, kappa=args.kappa, ablate_margin=args.ablate_margin
    )
    if report["reason"] == _ONLY_1D_REASON:
        print(
            "assist zoning-report: only 1D outputs are supported in this milestone",
            file=sys.stderr,
        )
        return 2

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as stream:
                stream.write(rendered)
        except OSError as error:
            print(
                "assist zoning-report: cannot write output: {0}".format(error),
                file=sys.stderr,
            )
            return 2
    else:
        print(rendered, end="")

    if args.strict and report["lint_a"] is not None:
        if report["lint_a"]["ok"] is False:
            return 2
    return 0


def main_promote_zoning(args) -> int:
    """CLI entry point for the promote-zoning verb."""
    payload, exit_code = promote_zoning(
        args.output_dir, kappa=args.kappa, margin_cells=args.margin_cells
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code
