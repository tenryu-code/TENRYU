#!/usr/bin/env python3
"""Generate, run, analyze, and report the 1D mesh-convergence campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence


UM, NS = 1.0e-4, 1.0e-9
M_FOIL = 2.6e-3                # g/cm^2 (25 um of solid CD)
RHO_SOLID = 1.05
A_INT = 5.0e-6                 # interior cell areal mass [g/cm^2]
A_SURF_LADDER = [3.2e-5 * 2.0 ** (-k) for k in range(9)]
VOID_RHO = 1.0e-9
VOID_THICKNESS = 100.0 * UM
SHELL_R_OUT, SHELL_THICKNESS, SHELL_R_MAX, SHELL_FILL_RHO = 250.0 * UM, 7.0 * UM, 400.0 * UM, 0.02
TOLERANCES = {"P_a": 0.05, "m_abl": 0.10, "E_abs": 0.03, "rho_R": 0.10, "t_bo": 0.02}
VERDICT_OBSERVABLES = ("P_a", "m_abl", "E_abs", "t_bo")
T_START_WINDOW = 0.1 * NS


WAVEFORMS = {
    "W1": {
        "t_end_s": 1.2 * NS,
        "peak_W_cm2": 1.0e14,
        "describe": "1e14 W/cm^2, 100 ps rise, flat",
        "power_source": """def laser_intensity(t_s):
    if t_s <= 0.0:
        return 0.0
    if t_s < 0.10 * ns:
        return 1.0e14 * t_s / (0.10 * ns)
    if t_s <= 1.20 * ns:
        return 1.0e14
    return 0.0""",
    },
    "W2": {
        "t_end_s": 1.2 * NS,
        "peak_W_cm2": 3.0e14,
        "describe": "3e14 W/cm^2, 100 ps rise, flat",
        "power_source": """def laser_intensity(t_s):
    if t_s <= 0.0:
        return 0.0
    if t_s < 0.10 * ns:
        return 3.0e14 * t_s / (0.10 * ns)
    if t_s <= 1.20 * ns:
        return 3.0e14
    return 0.0""",
    },
    "W3": {
        "t_end_s": 1.2 * NS,
        "peak_W_cm2": 1.0e15,
        "describe": "1e15 W/cm^2, 100 ps rise, flat",
        "power_source": """def laser_intensity(t_s):
    if t_s <= 0.0:
        return 0.0
    if t_s < 0.10 * ns:
        return 1.0e15 * t_s / (0.10 * ns)
    if t_s <= 1.20 * ns:
        return 1.0e15
    return 0.0""",
    },
    "W4": {
        "t_end_s": 2.4 * NS,
        "peak_W_cm2": 3.0e14,
        "describe": "3e14 W/cm^2 Gaussian, 1.0 ns FWHM at 1.2 ns",
        "power_source": """def laser_intensity(t_s):
    if t_s < 0.0 or t_s > 2.40 * ns:
        return 0.0
    return 3.0e14 * exp(-4.0 * log(2.0) * ((t_s - 1.20 * ns) / (1.00 * ns)) ** 2)""",
    },
    "W5": {
        "t_end_s": 2.2 * NS,
        "peak_W_cm2": 3.0e14,
        "describe": "2e13 W/cm^2 foot then 3e14 W/cm^2 main",
        "power_source": """def laser_intensity(t_s):
    if t_s <= 0.0:
        return 0.0
    if t_s < 0.05 * ns:
        return 2.0e13 * t_s / (0.05 * ns)
    if t_s <= 1.00 * ns:
        return 2.0e13
    if t_s < 1.10 * ns:
        return 2.0e13 + (3.0e14 - 2.0e13) * (t_s - 1.00 * ns) / (0.10 * ns)
    if t_s <= 2.20 * ns:
        return 3.0e14
    return 0.0""",
    },
    "W6": {
        "t_end_s": 1.8 * NS,
        "peak_W_cm2": 5.0e14,
        "describe": "100 ps Gaussian picket then 3e14 W/cm^2 main",
        "power_source": """def laser_intensity(t_s):
    if t_s < 0.0:
        return 0.0
    if t_s < 0.35 * ns:
        return 5.0e14 * exp(-4.0 * log(2.0) * ((t_s - 0.15 * ns) / (0.10 * ns)) ** 2)
    if t_s < 0.60 * ns:
        return 0.0
    if t_s < 0.70 * ns:
        return 3.0e14 * (t_s - 0.60 * ns) / (0.10 * ns)
    if t_s <= 1.60 * ns:
        return 3.0e14
    return 0.0""",
    },
    "W7": {
        "t_end_s": 2.1 * NS,
        "peak_W_cm2": 5.0e14,
        "describe": "linear ramp to 5e14 W/cm^2 over 2.0 ns",
        "power_source": """def laser_intensity(t_s):
    if t_s <= 0.0:
        return 0.0
    if t_s <= 2.00 * ns:
        return 5.0e14 * t_s / (2.00 * ns)
    return 0.0""",
    },
    "W8": {
        "t_end_s": 3.2 * NS,
        "peak_W_cm2": 3.0e13,
        "describe": "3e13 W/cm^2, 100 ps rise, long flat",
        "power_source": """def laser_intensity(t_s):
    if t_s <= 0.0:
        return 0.0
    if t_s < 0.10 * ns:
        return 3.0e13 * t_s / (0.10 * ns)
    if t_s <= 3.20 * ns:
        return 3.0e13
    return 0.0""",
    },
}


def _layer(name: str, rho: float, thickness_cm: float) -> dict[str, Any]:
    return {"name": name, "rho": rho, "thickness_cm": thickness_cm}


def _case(
    case_id: str,
    geometry: str,
    wavelength_nm: int,
    waveform: str,
    layers: list[dict[str, Any]],
    describe: str,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "geometry": geometry,
        "wavelength_nm": wavelength_nm,
        "waveform": waveform,
        "layers": layers,
        "describe": describe,
    }


_SOLID_FOIL = [_layer("solid", RHO_SOLID, M_FOIL / RHO_SOLID)]
CASES = [
    _case(
        "C{0:02d}".format(index),
        "planar",
        wavelength,
        waveform,
        [dict(_SOLID_FOIL[0])],
        "planar solid CD foil",
    )
    for index, (wavelength, waveform) in enumerate(
        (
            (351, "W1"), (351, "W2"), (351, "W3"),
            (527, "W1"), (527, "W2"), (527, "W3"),
            (1053, "W1"), (1053, "W2"), (1053, "W3"),
        ),
        start=1,
    )
]
CASES.extend(
    _case(
        "C{0:02d}".format(index),
        "planar",
        351,
        waveform,
        [dict(_SOLID_FOIL[0])],
        "planar solid CD foil, {0}".format(WAVEFORMS[waveform]["describe"]),
    )
    for index, waveform in enumerate(("W4", "W5", "W6", "W7", "W8"), start=10)
)
for index, rho in enumerate((0.05, 0.2, 0.5, 2.5), start=15):
    CASES.append(
        _case(
            "C{0:02d}".format(index),
            "planar",
            351,
            "W2",
            [_layer("CD", rho, M_FOIL / rho)],
            "planar CD density variant rho={0:g} g/cc".format(rho),
        )
    )
for index, rho in enumerate((0.2, 2.5), start=19):
    CASES.append(
        _case(
            "C{0:02d}".format(index),
            "planar",
            1053,
            "W2",
            [_layer("CD", rho, M_FOIL / rho)],
            "planar CD density variant rho={0:g} g/cc".format(rho),
        )
    )
CASES.extend(
    [
        _case(
            "C21", "planar", 351, "W2",
            [_layer("solid", RHO_SOLID, 10.0 * UM)],
            "planar solid CD, 10 um",
        ),
        _case(
            "C22", "planar", 351, "W2",
            [_layer("solid", RHO_SOLID, 50.0 * UM)],
            "planar solid CD, 50 um",
        ),
        _case(
            "C23", "planar", 351, "W2",
            [
                _layer("solid", RHO_SOLID, 20.0 * UM),
                _layer("foam", 0.1, 25.0 * UM),
            ],
            "planar foam-on-solid CD",
        ),
        _case(
            "C24", "planar", 351, "W2",
            [
                _layer("foam", 0.1, 50.0 * UM),
                _layer("solid", RHO_SOLID, 10.0 * UM),
            ],
            "planar solid-on-foam CD",
        ),
    ]
)
for index, rho in enumerate((0.2, 2.5), start=25):
    CASES.append(
        _case(
            "C{0:02d}".format(index),
            "planar",
            351,
            "W5",
            [_layer("CD", rho, M_FOIL / rho)],
            "planar CD density variant rho={0:g} g/cc, foot drive".format(rho),
        )
    )
CASES.append(
    _case(
        "C27", "planar", 351, "W6",
        [_layer("CD", 0.2, M_FOIL / 0.2)],
        "planar CD rho=0.2 g/cc, picket drive",
    )
)
_SHELL_LAYERS = [
    _layer("fill", SHELL_FILL_RHO, SHELL_R_OUT - SHELL_THICKNESS),
    _layer("shell", RHO_SOLID, SHELL_THICKNESS),
]
CASES.extend(
    [
        _case(
            "C28", "spherical", 527, "W4",
            [dict(layer) for layer in _SHELL_LAYERS],
            "spherical GXII CD shell, Gaussian drive",
        ),
        _case(
            "C29", "spherical", 351, "W3",
            [dict(layer) for layer in _SHELL_LAYERS],
            "spherical GXII CD shell, high-intensity drive",
        ),
    ]
)
CASE_BY_ID = {case["id"]: case for case in CASES}
_SANITY_BASE = CASE_BY_ID["C02"]
SANITY_CASES = [
    {
        **_SANITY_BASE,
        "id": "C02-S1",
        "layers": [dict(layer) for layer in _SANITY_BASE["layers"]],
        "describe": (
            "C02 interior-resolution sanity study at fixed surface areal mass "
            "2e-6 g/cm^2"
        ),
        "surface_ladder": [2.0e-6] * 4,
        "interior_ladder": [2.0e-5, 1.0e-5, 5.0e-6, 2.5e-6],
        "cfl_ladder": [0.3] * 4,
        "ladder_kind": "interior",
    },
    {
        **_SANITY_BASE,
        "id": "C02-S2",
        "layers": [dict(layer) for layer in _SANITY_BASE["layers"]],
        "describe": "C02 CFL sensitivity at the finest surface level",
        "surface_ladder": [5.0e-7] * 2,
        "interior_ladder": [5.0e-6] * 2,
        "cfl_ladder": [0.3, 0.15],
        "ladder_kind": "cfl",
    },
]
CASE_BY_ID.update({case["id"]: case for case in SANITY_CASES})
REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_RE = re.compile(r"_[0-9]{4,}\.h5$")


class CampaignError(RuntimeError):
    """A command-line configuration or campaign-data error."""


def waveform_intensity(waveform_id: str, t_s: float) -> float:
    """Evaluate one of the prescribed intensity waveforms in W/cm^2."""
    if waveform_id in ("W1", "W2", "W3"):
        peak = float(WAVEFORMS[waveform_id]["peak_W_cm2"])
        if t_s <= 0.0:
            return 0.0
        if t_s < 0.10 * NS:
            return peak * t_s / (0.10 * NS)
        if t_s <= 1.20 * NS:
            return peak
        return 0.0
    if waveform_id == "W4":
        if t_s < 0.0 or t_s > 2.40 * NS:
            return 0.0
        return 3.0e14 * math.exp(
            -4.0 * math.log(2.0) * ((t_s - 1.20 * NS) / (1.00 * NS)) ** 2
        )
    if waveform_id == "W5":
        if t_s <= 0.0:
            return 0.0
        if t_s < 0.05 * NS:
            return 2.0e13 * t_s / (0.05 * NS)
        if t_s <= 1.00 * NS:
            return 2.0e13
        if t_s < 1.10 * NS:
            return 2.0e13 + (3.0e14 - 2.0e13) * (t_s - 1.00 * NS) / (0.10 * NS)
        if t_s <= 2.20 * NS:
            return 3.0e14
        return 0.0
    if waveform_id == "W6":
        if t_s < 0.0:
            return 0.0
        if t_s < 0.35 * NS:
            return 5.0e14 * math.exp(
                -4.0 * math.log(2.0) * ((t_s - 0.15 * NS) / (0.10 * NS)) ** 2
            )
        if t_s < 0.60 * NS:
            return 0.0
        if t_s < 0.70 * NS:
            return 3.0e14 * (t_s - 0.60 * NS) / (0.10 * NS)
        if t_s <= 1.60 * NS:
            return 3.0e14
        return 0.0
    if waveform_id == "W7":
        if t_s <= 0.0:
            return 0.0
        if t_s <= 2.00 * NS:
            return 5.0e14 * t_s / (2.00 * NS)
        return 0.0
    if waveform_id == "W8":
        if t_s <= 0.0:
            return 0.0
        if t_s < 0.10 * NS:
            return 3.0e13 * t_s / (0.10 * NS)
        if t_s <= 3.20 * NS:
            return 3.0e13
        return 0.0
    raise CampaignError("unknown waveform: {0}".format(waveform_id))


def layer_edges(case: dict[str, Any]) -> list[float]:
    edges: list[float] = []
    position = 0.0
    for layer in case["layers"]:
        position += float(layer["thickness_cm"])
        edges.append(position)
    return edges


def _planar_mass_coordinate(case: dict[str, Any], radius: float) -> float:
    mass = 0.0
    start = 0.0
    for layer, edge in zip(case["layers"], layer_edges(case)):
        width = max(0.0, min(radius, edge) - start)
        mass += float(layer["rho"]) * width
        start = edge
        if radius <= edge:
            break
    return mass


def _planar_radius_at_mass(case: dict[str, Any], wanted_mass: float) -> float:
    accumulated = 0.0
    start = 0.0
    for layer, edge in zip(case["layers"], layer_edges(case)):
        layer_mass = float(layer["rho"]) * (edge - start)
        if wanted_mass <= accumulated + layer_mass:
            return start + (wanted_mass - accumulated) / float(layer["rho"])
        accumulated += layer_mass
        start = edge
    return layer_edges(case)[-1]


def _density_at(case: dict[str, Any], radius: float) -> float:
    for layer, edge in zip(case["layers"], layer_edges(case)):
        if radius < edge:
            return float(layer["rho"])
    return VOID_RHO


def _trapezoid_mass_table(case: dict[str, Any]) -> tuple[list[float], list[float]]:
    n_points = 4096
    r_max = SHELL_R_MAX
    radii = [r_max * index / float(n_points - 1) for index in range(n_points)]
    integrand = [
        4.0 * math.pi * radius * radius * _density_at(case, radius)
        for radius in radii
    ]
    cumulative = [0.0]
    for index in range(1, n_points):
        dr = radii[index] - radii[index - 1]
        cumulative.append(
            cumulative[-1] + 0.5 * (integrand[index - 1] + integrand[index]) * dr
        )
    return radii, cumulative


def _linear_interp(x: float, xp: Sequence[float], fp: Sequence[float]) -> float:
    if x <= xp[0]:
        return float(fp[0])
    if x >= xp[-1]:
        return float(fp[-1])
    lo = 0
    hi = len(xp) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xp[mid] <= x:
            lo = mid
        else:
            hi = mid
    weight = (x - xp[lo]) / (xp[hi] - xp[lo])
    return float(fp[lo] + weight * (fp[hi] - fp[lo]))


def case_levels(case: dict[str, Any]) -> list[int]:
    """Return the valid level indices for a campaign case."""
    surface_ladder = case.get("surface_ladder", A_SURF_LADDER)
    return list(range(len(surface_ladder)))


def _case_ladder(
    case: dict[str, Any],
    key: str,
    default_value: float,
) -> Sequence[float]:
    surface_ladder = case.get("surface_ladder", A_SURF_LADDER)
    ladder = case.get(key, [default_value] * len(surface_ladder))
    if len(ladder) != len(surface_ladder):
        raise CampaignError(
            "{0} must have {1} values for {2}".format(
                key, len(surface_ladder), case["id"]
            )
        )
    return ladder


def _validate_level(case: dict[str, Any], level: int) -> None:
    levels = case_levels(case)
    if level not in levels:
        raise CampaignError(
            "level must be between 0 and {0} for {1}".format(
                len(levels) - 1, case["id"]
            )
        )


def band_spec(
    case: dict[str, Any],
    level: int,
    surface_areal_mass: float | None = None,
    interior_areal_mass: float | None = None,
) -> dict[str, Any]:
    """Return physical band boundaries and measure-space ceilings."""
    _validate_level(case, level)
    surface_ladder = case.get("surface_ladder", A_SURF_LADDER)
    interior_ladder = _case_ladder(case, "interior_ladder", A_INT)
    a_surface = surface_ladder[level] if surface_areal_mass is None else surface_areal_mass
    a_inner = interior_ladder[level] if interior_areal_mass is None else interior_areal_mass
    if case["geometry"] == "planar":
        total_mass = sum(
            float(layer["rho"]) * float(layer["thickness_cm"])
            for layer in case["layers"]
        )
        midpoint = _planar_radius_at_mass(case, 0.5 * total_mass)
        fraction = _planar_mass_coordinate(case, midpoint) / total_mass
        return {
            "r_inner": 0.0,
            "r_mid": midpoint,
            "r_outer": layer_edges(case)[-1],
            "f_lo_inner": 0.0,
            "f_lo_outer": fraction,
            "f_hi_outer": 1.0,
            "outer_cell_measure_max": 1.05 * a_surface,
            "inner_cell_measure_max": 1.05 * a_inner,
        }

    r_in = SHELL_R_OUT - SHELL_THICKNESS
    radii, cumulative = _trapezoid_mass_table(case)
    mass_in = _linear_interp(r_in, radii, cumulative)
    mass_out = _linear_interp(SHELL_R_OUT, radii, cumulative)
    shell_mid_mass = mass_in + 0.5 * (mass_out - mass_in)
    r_mid = _linear_interp(shell_mid_mass, cumulative, radii)
    total_mass = cumulative[-1]
    return {
        "r_inner": r_in,
        "r_mid": r_mid,
        "r_outer": SHELL_R_OUT,
        "f_lo_inner": mass_in / total_mass,
        "f_lo_outer": shell_mid_mass / total_mass,
        "f_hi_outer": mass_out / total_mass,
        "outer_cell_measure_max": 4.0 * math.pi * r_mid * r_mid * 1.05 * a_surface,
        "inner_cell_measure_max": 4.0 * math.pi * r_in * r_in * 1.05 * a_inner,
    }


def level_budget(
    case: dict[str, Any],
    level: int,
    surface_areal_mass: float | None = None,
    interior_areal_mass: float | None = None,
    margin: float = 1.10,
) -> dict[str, Any]:
    """Compute the prescribed cell budget for one level."""
    _validate_level(case, level)
    surface_ladder = case.get("surface_ladder", A_SURF_LADDER)
    interior_ladder = _case_ladder(case, "interior_ladder", A_INT)
    a_surface = surface_ladder[level] if surface_areal_mass is None else surface_areal_mass
    a_inner = interior_ladder[level] if interior_areal_mass is None else interior_areal_mass
    bands = band_spec(case, level, surface_areal_mass, interior_areal_mass)
    midpoint = float(bands["r_mid"])
    if case["geometry"] == "planar":
        target_mass = sum(
            float(layer["rho"]) * float(layer["thickness_cm"])
            for layer in case["layers"]
        )
        mass_outer = 0.5 * target_mass
        mass_inner = 0.5 * target_mass
        material_start = 0.0
        n_fill = 0
    else:
        r_in = SHELL_R_OUT - SHELL_THICKNESS
        shell_mass = (
            (4.0 / 3.0)
            * math.pi
            * RHO_SOLID
            * (SHELL_R_OUT ** 3 - r_in ** 3)
        )
        mass_outer = 0.5 * shell_mass / (4.0 * math.pi * SHELL_R_OUT ** 2)
        mass_inner = mass_outer
        material_start = r_in
        n_fill = 40
    n_outer = int(math.ceil(mass_outer / a_surface))
    n_inner = int(math.ceil(mass_inner / a_inner))
    n_void = 40
    n_feather = 2 * int(
        math.ceil(abs(math.log(a_surface / a_inner)) / math.log(1.3))
    )
    n_required_per_layer: list[int] = []
    n_thin_extra = 0.0
    previous_n_thin_extra = 0.0
    layer_start = 0.0
    for layer, layer_end in zip(case["layers"], layer_edges(case)):
        segment_start = max(layer_start, material_start)
        if layer_end > segment_start:
            rho = float(layer["rho"])
            inner_width = max(0.0, min(layer_end, midpoint) - segment_start)
            outer_width = max(0.0, layer_end - max(segment_start, midpoint))
            nominal = rho * inner_width / a_inner + rho * outer_width / a_surface
            if nominal < 40.0:
                previous_n_thin_extra += 40.0 - nominal
            if case["geometry"] == "planar":
                mass_inner_layer = rho * inner_width
                mass_outer_layer = rho * outer_width
                inner_ceiling = 1.05 * a_inner
                outer_ceiling = 1.05 * a_surface
            else:
                inner_end = min(layer_end, midpoint)
                outer_start = max(segment_start, midpoint)
                mass_inner_layer = (
                    (4.0 / 3.0)
                    * math.pi
                    * rho
                    * max(0.0, inner_end ** 3 - segment_start ** 3)
                )
                mass_outer_layer = (
                    (4.0 / 3.0)
                    * math.pi
                    * rho
                    * max(0.0, layer_end ** 3 - outer_start ** 3)
                )
                inner_ceiling = float(bands["inner_cell_measure_max"])
                outer_ceiling = float(bands["outer_cell_measure_max"])
            required = (
                int(math.ceil(mass_outer_layer / outer_ceiling))
                + int(math.ceil(mass_inner_layer / inner_ceiling))
                + 2
            )
            if required < 40:
                n_thin_extra += 40 - required
                required = 40
            n_required_per_layer.append(required)
        layer_start = layer_end
    n_cells = max(
        int(math.ceil(margin * (
            sum(n_required_per_layer)
            + n_void
            + n_fill
            + n_feather
        ))),
        int(math.ceil(margin * (
            n_outer
            + n_inner
            + n_void
            + n_fill
            + n_feather
            + previous_n_thin_extra
        ))),
    )
    return {
        "a_surface_g_cm2": a_surface,
        "a_inner_g_cm2": a_inner,
        "M_outer_g_cm2": mass_outer,
        "M_inner_g_cm2": mass_inner,
        "n_outer": n_outer,
        "n_inner": n_inner,
        "n_void": n_void,
        "n_fill": n_fill,
        "n_feather": n_feather,
        "n_thin_extra": n_thin_extra,
        "n_required_per_layer": n_required_per_layer,
        "n_cells": n_cells,
    }


def _python_float(value: float) -> str:
    return repr(float(value))


def render_deck(
    case: dict[str, Any],
    level: int,
    root: Path,
    surface_areal_mass: float | None = None,
    interior_areal_mass: float | None = None,
    cfl_hydro: float | None = None,
    margin: float = 1.10,
) -> str:
    """Render one generated namelist without importing TENRYU."""
    _validate_level(case, level)
    cfl_ladder = _case_ladder(case, "cfl_ladder", 0.3)
    selected_cfl = cfl_ladder[level] if cfl_hydro is None else cfl_hydro
    waveform = WAVEFORMS[case["waveform"]]
    edges = layer_edges(case)
    densities = [float(layer["rho"]) for layer in case["layers"]]
    bands = band_spec(case, level, surface_areal_mass, interior_areal_mass)
    budget = level_budget(
        case, level, surface_areal_mass, interior_areal_mass, margin
    )
    if case["geometry"] == "planar":
        profile_rows = [
            {"r": 0.0, "w": budget["a_inner_g_cm2"]},
            {
                "r": bands["r_mid"] * (1.0 - 1.0e-6),
                "w": budget["a_inner_g_cm2"],
            },
            {"r": bands["r_mid"], "w": budget["a_surface_g_cm2"]},
            {"r": bands["r_outer"], "w": budget["a_surface_g_cm2"]},
        ]
    else:
        r_in = float(bands["r_inner"])
        r_mid = float(bands["r_mid"])
        r_out = float(bands["r_outer"])
        fill_rho = float(case["layers"][0]["rho"])
        fill_mass = (4.0 / 3.0) * math.pi * fill_rho * r_in ** 3
        profile_rows = [
            {"r": r_in * 1.0e-3, "w": fill_mass / 40.0},
            {"r": r_in * (1.0 - 1.0e-6), "w": fill_mass / 40.0},
            {
                "r": r_in,
                "w": 4.0 * math.pi * r_in ** 2 * budget["a_inner_g_cm2"],
            },
            {
                "r": r_mid * (1.0 - 1.0e-6),
                "w": 4.0 * math.pi * r_mid ** 2 * budget["a_inner_g_cm2"],
            },
            {
                "r": r_mid,
                "w": 4.0 * math.pi * r_mid ** 2 * budget["a_surface_g_cm2"],
            },
            {
                "r": r_out,
                "w": 4.0 * math.pi * r_out ** 2 * budget["a_surface_g_cm2"],
            },
        ]
    profile_literal = "[" + ", ".join(
        '{{"r": {0}, "w": {1}}}'.format(
            _python_float(row["r"]), _python_float(row["w"])
        )
        for row in profile_rows
    ) + "]"
    band_rows = [
        {
            "measure_frac_begin": bands["f_lo_outer"],
            "measure_frac_end": bands["f_hi_outer"],
            "cell_measure_max": bands["outer_cell_measure_max"],
        },
        {
            "measure_frac_begin": bands["f_lo_inner"],
            "measure_frac_end": bands["f_lo_outer"],
            "cell_measure_max": bands["inner_cell_measure_max"],
        },
    ]
    output_dir = root / "runs" / case["id"] / "L{0}".format(level)
    direction = "(-1.0, 0.0, 0.0)" if case["geometry"] == "planar" else "(0.0, 0.0, -1.0)"
    focus = "(0.0, 0.0, 0.0)" if case["geometry"] == "planar" else "(0.0, 0.0, -750.0e-4)"
    area = "1.0" if case["geometry"] == "planar" else "4.0 * pi * R_TARGET ** 2"
    r_max_definition = (
        "R_MAX = R_TARGET + VOID_THICKNESS"
        if case["geometry"] == "planar"
        else "SHELL_R_MAX = 400.0 * um\nR_MAX = SHELL_R_MAX"
    )
    return """from tenryu_namelist import *
import numpy as np
from math import exp, log, sqrt, pi

um = 1.0e-4
ns = 1.0e-9

# --- case constants (generated) ---
CASE_ID = {case_id!r}
WAVEFORM_ID = {waveform_id!r}
LAYER_EDGES = {edges!r}
LAYER_RHO = {densities!r}
R_MIN = 0.0
R_TARGET = LAYER_EDGES[-1]
VOID_RHO = 1.0e-9
VOID_THICKNESS = 100.0 * um
{r_max_definition}

mat_cd = Material(
    name="CD", A=7.0, Z=3.5,
    eos=dict(model="tmat", file="TMAT-H5/CD.tmat.h5"),
    opacity=dict(
        model="tmat", file="TMAT-H5/CD.tmat.h5",
        lambda_method="finite_difference", lambda_fd_delta_rel=1.0e-4,
        lambda_fd_abs_min=1.0e-6, f_min=1.0e-4,
    ),
)
mat_void = Material(name="VOID", A=1.0, Z=1.0, is_void=True)

Main(
    name={main_name!r}, dimension="1D_SPH", temperature_model="2T",
    t_end={t_end}, seed=12345, max_steps=10_000_000, verbosity="normal",
)

Mesh(
    r_min=R_MIN, r_max=R_MAX, geometry_1d={geometry!r}, motion="lagrangian",
    zoning_intent=dict(
        n_cells={n_cells}, measure={measure!r},
        density_regions=[
            {{"r_end": edge, "rho": rho}}
            for edge, rho in zip(LAYER_EDGES, LAYER_RHO)
        ] + [{{"r_end": R_MAX, "rho": VOID_RHO}}],
        pins=[
            {{"r": edge, "ratio_jump_allowed": True}}
            for edge in LAYER_EDGES
        ],
        profile={profile_literal},
        bands={band_rows!r},
        ratio_hard_max=1.3, min_cells_per_segment=40,
    ),
    floors=dict(rho_floor_gcc=1.0e-9, Te_floor_eV=0.1, Ti_floor_eV=0.1),
)

Materials(
    materials=[mat_cd, mat_void], opacity_mix_rule="linear_mass",
    zbar=dict(model="fixed", fixed_value=3.5),
    void_config=dict(rho=1.0e-9, Te=1.0, Ti=1.0),
)

def vf_cd(x_cm, z_cm=0.0):
    return np.where(x_cm < R_TARGET, 1.0, 0.0)

def vf_void(x_cm, z_cm=0.0):
    return np.where(x_cm < R_TARGET, 0.0, 1.0)

def rho_profile(x_cm, z_cm=0.0):
    value = np.where(x_cm < LAYER_EDGES[-1], LAYER_RHO[-1], VOID_RHO)
    for edge, rho in reversed(list(zip(LAYER_EDGES[:-1], LAYER_RHO[:-1]))):
        value = np.where(x_cm < edge, rho, value)
    return value

def Te_profile(x_cm, z_cm=0.0):
    return 1.0

def Ti_profile(x_cm, z_cm=0.0):
    return 1.0

Geometry(
    rho=rho_profile, Te=Te_profile, Ti=Ti_profile,
    volfrac=dict(CD=vf_cd, VOID=vf_void), enforce_sum_to_one=True,
)

Radiation(enabled=False)

# Waveform {waveform_id}
{power_source}

AREA_CM2 = {area}

def laser_power(t_s):
    return laser_intensity(t_s) * AREA_CM2

Laser(
    enabled=True, wavelength_nm={wavelength}, mode="radial_absorption_1d",
    rays_per_beam=8000, ray_output_count=0, ray_output_trajectory=False,
    absorption=dict(model="inverse_bremsstrahlung"),
    lasermesh=dict(
        mesh_factor=0.1, rmax_n_hat_threshold=0.001,
        ghost_corona=dict(
            enabled=True, n_out=12, ne_min_frac=0.03, ne_max_frac=0.99,
            Te_min_eV=50.0, zbar_min=1.0, zbar_max=4.0,
            handoff_cells=6, handoff_decay=2.0, transition_enabled=True,
            transition_resolved_nhat=0.9, transition_resolved_cells=3,
            transition_density_exponent=1.0,
        ),
    ),
    raytrace=dict(
        ds_adapt_g_target=0.05, ds_adapt_tau_target=0.05,
        ds_adapt_max_factor=2.0,
    ),
    deposit=dict(deposit_smooth_passes=3, deposit_smooth_alpha=0.25),
    beams=[
        LaserBeam(
            name="beam_00", direction={direction}, power=laser_power,
            f_number=3.0, focus={focus},
            profile=dict(model="super_gaussian", w0_um=250.0, m=4),
        )
    ],
    cbet=dict(enable=False), hot_electron=dict(enable=False),
)

Burn(enabled=False)

Numerics(
    dt=dict(
        initial_s=1.0e-13, max_s=1.0e-10,
        cfl_hydro={cfl_hydro}, cfl_cond=0.25,
    ),
    hydro=dict(boundary_1d="free", driver_full_step_retry_enabled=True),
    conduction=dict(enabled=True, solver="implicit", f_lim=0.06),
    positivity=dict(clamp=True), safety=dict(nan_fatal=True),
    diagnostics_every=100,
)

Output(
    directory={output_dir!r}, format="hdf5", plot_every_s=1.0e-11,
    history_every_s=1.0e-11, checkpoint_every=0, checkpoint_keep_last=1,
    save_namelist_copy=True, save_frozen_config=True,
)

Diagnostics(
    enabled=True, every=1,
    energy_budget=dict(enabled=True, warn_threshold=1.0e-3),
)
""".format(
        case_id=case["id"],
        waveform_id=case["waveform"],
        edges=edges,
        densities=densities,
        r_max_definition=r_max_definition,
        main_name="{0}_L{1}".format(case["id"], level),
        t_end=_python_float(float(waveform["t_end_s"])),
        geometry=case["geometry"],
        n_cells=budget["n_cells"],
        measure="areal_mass" if case["geometry"] == "planar" else "spherical_cell_mass",
        profile_literal=profile_literal,
        band_rows=band_rows,
        power_source=waveform["power_source"],
        area=area,
        wavelength=_python_float(float(case["wavelength_nm"])),
        direction=direction,
        focus=focus,
        cfl_hydro=_python_float(selected_cfl),
        output_dir=str(output_dir),
    )


def _write_text_if_changed(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    _write_text_if_changed(path, json.dumps(payload, sort_keys=True, indent=2) + "\n")


def parse_levels(
    specification: str | None,
    case: dict[str, Any] | None = None,
) -> list[int]:
    selected_case = CASES[0] if case is None else case
    valid_levels = case_levels(selected_case)
    if specification is None:
        return valid_levels
    levels: set[int] = set()
    for field in specification.split(","):
        field = field.strip()
        if not field:
            continue
        if "-" in field:
            pieces = field.split("-", 1)
            try:
                first, last = int(pieces[0]), int(pieces[1])
            except ValueError as exc:
                raise CampaignError("invalid level range: {0}".format(field)) from exc
            if first > last:
                raise CampaignError("invalid descending level range: {0}".format(field))
            levels.update(range(first, last + 1))
        else:
            try:
                levels.add(int(field))
            except ValueError as exc:
                raise CampaignError("invalid level: {0}".format(field)) from exc
    if not levels or any(level not in valid_levels for level in levels):
        raise CampaignError(
            "levels must be in the range 0-{0} for {1}".format(
                len(valid_levels) - 1, selected_case["id"]
            )
        )
    return sorted(levels)


def select_cases(specification: str | None) -> list[dict[str, Any]]:
    if specification is None:
        return list(CASES)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case_id in (field.strip() for field in specification.split(",")):
        if not case_id or case_id in seen:
            continue
        if case_id not in CASE_BY_ID:
            raise CampaignError("unknown case: {0}".format(case_id))
        selected.append(CASE_BY_ID[case_id])
        seen.add(case_id)
    if not selected:
        raise CampaignError("at least one case must be selected")
    return selected


def _sanity_variants(root: Path) -> list[tuple[dict[str, Any], int, str]]:
    return [
        (case, level, render_deck(case, level, root))
        for case in SANITY_CASES
        for level in case_levels(case)
    ]


def generate_decks(
    root: Path,
    cases: Sequence[dict[str, Any]],
    levels: Sequence[int],
    sanity: bool = False,
) -> list[Path]:
    written: list[Path] = []
    for case in cases:
        for level in levels:
            path = root / "decks" / case["id"] / "L{0}.py".format(level)
            _write_text_if_changed(path, render_deck(case, level, root))
            _write_json(
                path.with_suffix(".budget.json"),
                {
                    "margin": 1.10,
                    "n_cells": level_budget(case, level)["n_cells"],
                },
            )
            written.append(path)
    if sanity:
        for case, level, text in _sanity_variants(root):
            path = root / "decks" / case["id"] / "L{0}.py".format(level)
            _write_text_if_changed(path, text)
            _write_json(
                path.with_suffix(".budget.json"),
                {
                    "margin": 1.10,
                    "n_cells": level_budget(case, level)["n_cells"],
                },
            )
            if path not in written:
                written.append(path)
    return written


def _termination_reason(run_dir: Path) -> str | None:
    path = run_dir / "run_info.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("termination_reason")
    return None if value is None else str(value)


def _is_done(run_dir: Path) -> tuple[bool, str | None]:
    reason = _termination_reason(run_dir)
    if reason is None:
        return False, None
    normalized = reason.lower()
    return (
        reason == "t_end" or "completed" in normalized or "t_end" in normalized,
        reason,
    )


def _wall_seconds(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    match = re.search(
        r"^mesh_convergence_wall_seconds=([0-9.eE+-]+)$",
        log_path.read_text(encoding="utf-8", errors="replace"),
        re.MULTILINE,
    )
    return None if match is None else float(match.group(1))


def _budget_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def run_campaign(
    root: Path,
    tenryu: Path,
    cases: Sequence[dict[str, Any]],
    adaptive: bool,
    max_level: int | None,
    resume: bool,
    dry_run: bool,
) -> int:
    failed = False
    for case in cases:
        levels = case_levels(case)
        if max_level is not None:
            if max_level not in levels:
                raise CampaignError(
                    "--max-level must be in the range 0-{0} for {1}".format(
                        len(levels) - 1, case["id"]
                    )
                )
            levels = [level for level in levels if level <= max_level]
        for level in levels:
            deck_path = root / "decks" / case["id"] / "L{0}.py".format(level)
            budget_path = deck_path.with_suffix(".budget.json")
            run_dir = root / "runs" / case["id"] / "L{0}".format(level)
            log_path = run_dir.with_suffix(".log")
            if not deck_path.is_file():
                raise CampaignError("missing generated deck: {0}".format(deck_path))
            done, reason = _is_done(run_dir)
            metadata = _budget_metadata(budget_path)
            margin = 1.10 if metadata is None else float(metadata["margin"])
            budget = level_budget(case, level, margin=margin)
            if done:
                status = "skipped ({0})".format(reason) if resume else "already done ({0})".format(reason)
                print(
                    "{0} L{1} n_cells={2} margin={3:.2f} status={4} wall_s={5}".format(
                        case["id"], level, budget["n_cells"], margin, status,
                        _wall_seconds(log_path),
                    )
                )
            else:
                command = [
                    str(tenryu), "run", str(deck_path), "--output-dir", str(run_dir)
                ]
                if dry_run:
                    print(
                        "{0} L{1} n_cells={2} margin={3:.2f} status=dry-run wall_s=0 command={4}".format(
                            case["id"], level, budget["n_cells"], margin,
                            shlex.join(command),
                        )
                    )
                    continue
                retry_margins = iter(
                    candidate
                    for candidate in (1.25, 1.45, 1.70)
                    if candidate > margin
                )
                while True:
                    if run_dir.exists():
                        shutil.rmtree(run_dir)
                    run_dir.parent.mkdir(parents=True, exist_ok=True)
                    start = time.monotonic()
                    with log_path.open("w", encoding="utf-8") as stream:
                        completed = subprocess.run(
                            command,
                            cwd=str(REPO_ROOT),
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            check=False,
                        )
                        wall = time.monotonic() - start
                        stream.write("\nmesh_convergence_wall_seconds={0:.9f}\n".format(wall))
                        stream.write("mesh_convergence_returncode={0}\n".format(completed.returncode))
                        stream.write("margin={0:.2f}\n".format(margin))
                    done, reason = _is_done(run_dir)
                    status = reason if done else "failed (returncode={0}, termination={1})".format(
                        completed.returncode, reason
                    )
                    print(
                        "{0} L{1} n_cells={2} margin={3:.2f} status={4} wall_s={5:.6f}".format(
                            case["id"], level, budget["n_cells"], margin,
                            status, wall,
                        )
                    )
                    if done:
                        break
                    log_text = log_path.read_text(
                        encoding="utf-8", errors="replace"
                    )
                    if "INFEASIBLE" not in log_text:
                        failed = True
                        break
                    try:
                        margin = next(retry_margins)
                    except StopIteration:
                        failed = True
                        break
                    budget = level_budget(case, level, margin=margin)
                    _write_text_if_changed(
                        deck_path, render_deck(case, level, root, margin=margin)
                    )
                    _write_json(
                        budget_path,
                        {"margin": margin, "n_cells": budget["n_cells"]},
                    )
                    if run_dir.exists():
                        shutil.rmtree(run_dir)
                    print(
                        "{0} L{1} retry margin={2:.2f} n_cells={3}".format(
                            case["id"], level, margin, budget["n_cells"]
                        )
                    )
                if not done:
                    break
            if adaptive and level >= 1 and not dry_run:
                analyzed = analyze_case(root, case, max_level=level)
                if analyzed["converged"]:
                    break
    return 1 if failed else 0


def _require_dataset(handle: Any, name: str) -> Any:
    if name not in handle:
        raise CampaignError("required HDF5 dataset missing: {0}".format(name))
    return handle[name]


def _read_requirement(run_dir: Path) -> tuple[dict[str, Any], float | None]:
    path = run_dir / "mesh_requirement.json"
    if not path.exists():
        return {}, None
    requirement = json.loads(path.read_text(encoding="utf-8"))
    inputs = requirement.get("inputs", {})
    ablator = inputs.get("ablator", {}) if isinstance(inputs, dict) else {}
    params = requirement.get("params", {})
    ablation = requirement.get("ablation", {})
    check = requirement.get("requirement_check", {})
    ablation_check = check.get("ablation", {}) if isinstance(check, dict) else {}
    shock_check = check.get("shock", {}) if isinstance(check, dict) else {}
    rho_c = ablator.get("rho_c_gcc") if isinstance(ablator, dict) else None
    rule_ratios = [
        rule.get("max_ratio")
        for rule in (ablation_check, shock_check)
        if isinstance(rule, dict) and rule.get("max_ratio") is not None
    ]
    prediction = {
        "rho_c_gcc": rho_c,
        "zones_per_scale_length": (
            params.get("zones_per_scale_length")
            if isinstance(params, dict) else None
        ),
        "intensity_correction_factor": (
            ablation.get("intensity_correction_factor")
            if isinstance(ablation, dict) else None
        ),
        "ceiling_formation_g_cm2": (
            ablation.get("ceiling_formation_g_cm2")
            if isinstance(ablation, dict) else None
        ),
        "bands": requirement.get("bands_recommended"),
        "level_0_verdict_max_ratio": max(rule_ratios) if rule_ratios else None,
        "requirement_check": check,
    }
    return prediction, None if rho_c is None else float(rho_c)


def read_run_observables(run_dir: Path, case: dict[str, Any]) -> dict[str, Any]:
    """Read observables for one run.

    P_a is max(Pe + Pi) over unablated cells in the outer half by initial
    cumulative target mass (the larger-index, laser-facing side). For planar
    runs, P_rear is Pe + Pi in cell 0.
    """
    import h5py
    import numpy as np

    results_dir = run_dir / "results"
    snapshots = sorted(
        path
        for path in results_dir.glob("*.h5")
        if SNAPSHOT_RE.search(path.name) and not path.name.endswith("_history.h5")
    )
    if not snapshots:
        raise CampaignError("no snapshots found in {0}".format(results_dir))
    prediction, rho_c = _read_requirement(run_dir)
    with h5py.File(str(snapshots[0]), "r") as handle:
        rho_initial = np.asarray(_require_dataset(handle, "hydro/rho")[()], dtype=float).reshape(-1)
        mass_initial = np.asarray(
            _require_dataset(handle, "hydro/mass")[()], dtype=float
        ).reshape(-1)
    if rho_initial.size != mass_initial.size:
        raise CampaignError("cell dataset size mismatch in {0}".format(snapshots[0]))
    target_indices = np.flatnonzero(rho_initial > 1.0e-3)
    if target_indices.size == 0:
        raise CampaignError("no non-void target cells found in {0}".format(snapshots[0]))
    target_mask = rho_initial > 1.0e-3
    target_mass = np.where(target_mask, mass_initial, 0.0)
    total_target_mass = float(np.sum(target_mass))
    mass_depth = np.cumsum(target_mass[::-1])[::-1] - target_mass
    outer_half = target_mask & (mass_depth < 0.5 * total_target_mass)
    a_surface_achieved = float(np.mean(mass_initial[target_indices[-10:]]))
    if case["geometry"] == "spherical":
        a_surface_achieved /= 4.0 * math.pi * SHELL_R_OUT ** 2

    snapshot_times: list[float] = []
    p_a: list[float] = []
    m_abl: list[float] | None = [] if rho_c is not None else None
    rho_r: list[float] = []
    rear_times: list[tuple[float, float]] = []
    spherical_area = 4.0 * math.pi * SHELL_R_OUT ** 2
    for path in snapshots:
        with h5py.File(str(path), "r") as handle:
            time_s = float(handle.attrs["t"])
            rho = np.asarray(_require_dataset(handle, "hydro/rho")[()], dtype=float).reshape(-1)
            pe = np.asarray(_require_dataset(handle, "hydro/Pe")[()], dtype=float).reshape(-1)
            pi = np.asarray(_require_dataset(handle, "hydro/Pi")[()], dtype=float).reshape(-1)
            mass = np.asarray(_require_dataset(handle, "hydro/mass")[()], dtype=float).reshape(-1)
            np.asarray(_require_dataset(handle, "mesh/x_r")[()], dtype=float)
            np.asarray(_require_dataset(handle, "mesh/v_r")[()], dtype=float)
            deposited = np.asarray(
                _require_dataset(handle, "laser/deposited_power")[()], dtype=float
            ).reshape(-1)
            if not (
                rho.size == rho_initial.size == pe.size == pi.size == mass.size == deposited.size
            ):
                raise CampaignError("cell dataset size mismatch in {0}".format(path))
            pressure = pe + pi
            unablated = rho >= 0.5 * rho_initial
            ablation_pressure_mask = unablated & outer_half
            p_a.append(
                float(np.max(pressure[ablation_pressure_mask]))
                if np.any(ablation_pressure_mask) else 0.0
            )
            if m_abl is not None:
                ablated_mass = float(np.sum(mass[rho < 1.2 * float(rho_c)]))
                if case["geometry"] == "spherical":
                    ablated_mass /= spherical_area
                m_abl.append(ablated_mass)
            total_deposition = float(np.sum(deposited))
            rho_r.append(
                0.0
                if total_deposition == 0.0
                else float(np.sum(deposited * rho) / total_deposition)
            )
            if case["geometry"] == "planar":
                rear_times.append((time_s, float(pressure[0])))
            snapshot_times.append(time_s)

    order = np.argsort(np.asarray(snapshot_times))
    ordered_times = np.asarray(snapshot_times)[order].tolist()

    history_key = None
    history_times: list[float] | None = None
    e_abs: list[float] | None = None
    histories = sorted(results_dir.glob("*_history.h5"))
    if histories:
        with h5py.File(str(histories[0]), "r") as handle:
            times = np.asarray(_require_dataset(handle, "t")[()], dtype=float).reshape(-1)
            for candidate in (
                "energy/laser_ra_deposited",
                "energy/laser_deposited",
                "laser/absorbed_total",
            ):
                if candidate not in handle:
                    continue
                values = np.asarray(handle[candidate][()], dtype=float).reshape(-1)
                if candidate == "energy/laser_ra_deposited" and (
                    values.size == 0 or values[-1] <= 0.0
                ):
                    continue
                if values.size != times.size:
                    raise CampaignError("history dataset size mismatch: {0}".format(candidate))
                history_key = candidate
                history_order = np.argsort(times)
                history_times = times[history_order].tolist()
                e_abs = values[history_order].tolist()
                break

    t_bo = None
    for time_s, rear_pressure in sorted(rear_times):
        if rear_pressure > 5.0e11:
            t_bo = time_s
            break
    return {
        "P_a": {"t_s": ordered_times, "values": np.asarray(p_a)[order].tolist()},
        "P_rear": None if case["geometry"] != "planar" else {
            "t_s": [time_s for time_s, _ in sorted(rear_times)],
            "values": [pressure for _, pressure in sorted(rear_times)],
        },
        "P_rear_max_dyn_cm2": (
            None if case["geometry"] != "planar"
            else max(pressure for _, pressure in rear_times)
        ),
        "m_abl": None if m_abl is None else {
            "t_s": ordered_times,
            "values": np.asarray(m_abl)[order].tolist(),
        },
        "rho_R": {"t_s": ordered_times, "values": np.asarray(rho_r)[order].tolist()},
        "E_abs": None if e_abs is None else {"t_s": history_times, "values": e_abs},
        "t_bo": t_bo,
        "E_abs_history_key": history_key,
        "requirement_prediction": prediction,
        "a_surface_achieved_g_cm2": a_surface_achieved,
    }


def _series_deviation(
    candidate: dict[str, Any] | None,
    reference: dict[str, Any] | None,
    window_end: float,
) -> float | None:
    import numpy as np

    if candidate is None or reference is None:
        return None
    candidate_t = np.asarray(candidate["t_s"], dtype=float)
    candidate_q = np.asarray(candidate["values"], dtype=float)
    reference_t = np.asarray(reference["t_s"], dtype=float)
    reference_q = np.asarray(reference["values"], dtype=float)
    if not candidate_t.size or not reference_t.size:
        return None
    count = int(math.floor((window_end - T_START_WINDOW) / 1.0e-11 + 1.0e-9)) + 1
    if count <= 0:
        return None
    grid = T_START_WINDOW + np.arange(count, dtype=float) * 1.0e-11
    candidate_interp = np.interp(grid, candidate_t, candidate_q)
    reference_interp = np.interp(grid, reference_t, reference_q)
    denominator = float(np.linalg.norm(reference_interp))
    numerator = float(np.linalg.norm(candidate_interp - reference_interp))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def deviation_table(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    t_end: float,
) -> dict[str, Any]:
    reference_bo = reference.get("t_bo")
    window_end = (
        t_end if reference_bo is None else min(t_end, float(reference_bo))
    )
    window_note = None
    if window_end - T_START_WINDOW < 0.2e-9:
        window_end = t_end
        window_note = "breakout too early; full window used"
    deviations = {
        name: _series_deviation(candidate.get(name), reference.get(name), window_end)
        for name in ("P_a", "m_abl", "E_abs", "rho_R")
    }
    deviations["window_end_s"] = window_end
    if window_note is not None:
        deviations["window_note"] = window_note
    candidate_bo = candidate.get("t_bo")
    if candidate_bo is None or reference_bo is None:
        deviations["t_bo"] = None
    elif reference_bo == 0.0:
        deviations["t_bo"] = 0.0 if candidate_bo == 0.0 else float("inf")
    else:
        deviations["t_bo"] = abs(float(candidate_bo) - float(reference_bo)) / abs(float(reference_bo))
    return deviations


def _reference_pressure_mask_empties(
    case: dict[str, Any],
    observations: dict[int, dict[str, Any]],
) -> bool:
    if not observations:
        return False
    reference = observations[max(observations)]
    pressure = reference.get("P_a")
    if pressure is None:
        return False
    t_end = float(WAVEFORMS[case["waveform"]]["t_end_s"])
    reference_bo = reference.get("t_bo")
    window_end = (
        t_end if reference_bo is None else min(t_end, float(reference_bo))
    )
    if window_end - T_START_WINDOW < 0.2e-9:
        window_end = t_end
    return any(
        T_START_WINDOW <= float(time_s) <= window_end and float(value) == 0.0
        for time_s, value in zip(pressure["t_s"], pressure["values"])
    )


def verdict_observables(
    case: dict[str, Any],
    observations: dict[int, dict[str, Any]],
) -> tuple[str, ...]:
    """Return the observables used for the convergence verdict."""
    if case["geometry"] == "spherical":
        names = ("m_abl", "E_abs")
    elif len(case["layers"]) > 1:
        names = ("m_abl", "E_abs", "t_bo")
    else:
        names = VERDICT_OBSERVABLES
    if _reference_pressure_mask_empties(case, observations):
        names = tuple(name for name in names if name != "P_a")
    return names


def _all_within(
    deviations: dict[str, Any],
    scale: float,
    names: Iterable[str] = VERDICT_OBSERVABLES,
) -> bool:
    available = [
        float(deviations[name]) <= scale * TOLERANCES[name]
        for name in names
        if deviations.get(name) is not None
    ]
    return bool(available) and all(available)


def analyze_case(
    root: Path,
    case: dict[str, Any],
    max_level: int = 8,
) -> dict[str, Any]:
    """Analyze one case against its finest completed level.

    ``a_conv`` is converged to within the tolerance of the finest run whose own
    uncertainty is the last increment.
    """
    observations: dict[int, dict[str, Any]] = {}
    levels: list[int] = []
    levels_in_progress: list[int] = []
    for level in (level for level in case_levels(case) if level <= max_level):
        run_dir = root / "runs" / case["id"] / "L{0}".format(level)
        if not (run_dir / "results").is_dir():
            continue
        if not _is_done(run_dir)[0]:
            levels_in_progress.append(level)
            continue
        observations[level] = read_run_observables(run_dir, case)
        levels.append(level)

    level_data: dict[str, Any] = {}
    deviations: dict[str, Any] = {}
    converged = False
    converged_strict = False
    a_conv = None
    a_conv_nominal = None
    a_conv_lenient = None
    a_conv_lenient_nominal = None
    a_conv_strict = None
    a_conv_strict_nominal = None
    reference_last_increment = None
    convergence_rate_estimate = None
    prediction: dict[str, Any] = {}
    case_verdict_observables = verdict_observables(case, observations)
    strict_verdict_observables = (*case_verdict_observables, "rho_R")
    notes = []
    if _reference_pressure_mask_empties(case, observations):
        notes.append(
            "P_a excluded: unablated outer-half mask empties within the window"
        )
    if levels:
        reference_level = levels[-1]
        reference = observations[reference_level]
        t_end = float(WAVEFORMS[case["waveform"]]["t_end_s"])
        for level in levels:
            run_dir = root / "runs" / case["id"] / "L{0}".format(level)
            budget_path = (
                root / "decks" / case["id"] / "L{0}.budget.json".format(level)
            )
            budget_metadata = _budget_metadata(budget_path)
            data = {
                "n_cells": (
                    level_budget(case, level)["n_cells"]
                    if budget_metadata is None
                    else budget_metadata.get(
                        "n_cells", level_budget(case, level)["n_cells"]
                    )
                ),
                "budget_margin": (
                    None
                    if budget_metadata is None
                    else budget_metadata.get("margin")
                ),
                "wall_time_s": _wall_seconds(run_dir.with_suffix(".log")),
                "termination_reason": _termination_reason(run_dir),
                "E_abs_history_key": observations[level]["E_abs_history_key"],
                "a_surface_achieved_g_cm2": observations[level][
                    "a_surface_achieved_g_cm2"
                ],
                "P_rear_max_dyn_cm2": observations[level][
                    "P_rear_max_dyn_cm2"
                ],
            }
            if case.get("ladder_kind") == "interior":
                data["interior_areal_mass_g_cm2"] = _case_ladder(
                    case, "interior_ladder", A_INT
                )[level]
            elif case.get("ladder_kind") == "cfl":
                data["cfl_hydro"] = _case_ladder(
                    case, "cfl_ladder", 0.3
                )[level]
            level_data[str(level)] = data
            deviations[str(level)] = deviation_table(
                observations[level], reference, t_end
            )
        if 0 in observations:
            prediction = observations[0]["requirement_prediction"]
        if len(levels) >= 2:
            reference_last_increment = deviations[str(levels[-2])]
            converged = _all_within(
                reference_last_increment,
                1.0,
                names=case_verdict_observables,
            )
            converged_strict = _all_within(
                reference_last_increment,
                1.0,
                names=strict_verdict_observables,
            )
        if len(levels) >= 3:
            previous_deviation = deviations[str(levels[-3])].get("P_a")
            last_deviation = reference_last_increment.get("P_a")
            if (
                previous_deviation is not None
                and last_deviation is not None
                and float(previous_deviation) != 0.0
            ):
                convergence_rate_estimate = (
                    float(last_deviation) / float(previous_deviation)
                )
        for level in levels:
            if _all_within(
                deviations[str(level)],
                1.0,
                names=case_verdict_observables,
            ):
                a_conv_lenient = observations[level][
                    "a_surface_achieved_g_cm2"
                ]
                a_conv_lenient_nominal = case.get(
                    "surface_ladder", A_SURF_LADDER
                )[level]
                break
        if converged:
            for level in levels:
                if all(
                    _all_within(
                        deviations[str(finer)],
                        1.0,
                        names=case_verdict_observables,
                    )
                    for finer in levels
                    if finer >= level
                ):
                    a_conv = observations[level]["a_surface_achieved_g_cm2"]
                    a_conv_nominal = case.get("surface_ladder", A_SURF_LADDER)[level]
                    break
        if converged_strict:
            for level in levels:
                if all(
                    _all_within(
                        deviations[str(finer)],
                        1.0,
                        names=strict_verdict_observables,
                    )
                    for finer in levels
                    if finer >= level
                ):
                    a_conv_strict = observations[level][
                        "a_surface_achieved_g_cm2"
                    ]
                    a_conv_strict_nominal = case.get(
                        "surface_ladder", A_SURF_LADDER
                    )[level]
                    break
    ceiling = prediction.get("ceiling_formation_g_cm2") if prediction else None
    r_c = (
        None
        if a_conv is None or ceiling is None or float(ceiling) == 0.0
        else a_conv / float(ceiling)
    )
    r_c_strict = (
        None
        if a_conv_strict is None or ceiling is None or float(ceiling) == 0.0
        else a_conv_strict / float(ceiling)
    )
    result = {
        "id": case["id"],
        "describe": case["describe"],
        "geometry": case["geometry"],
        "wavelength_nm": case["wavelength_nm"],
        "waveform": case["waveform"],
        "layers": case["layers"],
        "levels_run": levels,
        "levels_in_progress": levels_in_progress,
        "level_data": level_data,
        "deviations": deviations,
        "reference_last_increment": reference_last_increment,
        "convergence_rate_estimate": convergence_rate_estimate,
        "verdict_observables": list(case_verdict_observables),
        "notes": notes,
        "converged": converged,
        "converged_strict": converged_strict,
        "a_conv_g_cm2": a_conv,
        "a_conv_nominal_g_cm2": a_conv_nominal,
        "a_conv_lenient_g_cm2": a_conv_lenient,
        "a_conv_lenient_nominal_g_cm2": a_conv_lenient_nominal,
        "monotone": a_conv_lenient == a_conv,
        "a_conv_strict_g_cm2": a_conv_strict,
        "a_conv_strict_nominal_g_cm2": a_conv_strict_nominal,
        "requirement_prediction": prediction,
        "r_c": r_c,
        "r_c_strict": r_c_strict,
    }
    if case.get("ladder_kind") in ("interior", "cfl"):
        result["ladder_kind"] = case["ladder_kind"]
    return result


def analyze_campaign(
    root: Path,
    cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "tenryu.mesh_convergence.v1",
        "cases": {
            case["id"]: analyze_case(root, case)
            for case in cases
        },
    }


def _format_float(value: Any, digits: int = 5) -> str:
    if value is None:
        return "—"
    return ("{0:." + str(digits) + "g}").format(float(value))


def _profile_label(layers: Sequence[dict[str, Any]]) -> str:
    return "; ".join(
        "{0}:{1:g} g/cc x {2:.6g} um".format(
            layer["name"], float(layer["rho"]), float(layer["thickness_cm"]) / UM
        )
        for layer in layers
    )


def _geometric_mean(values: Iterable[float]) -> float | None:
    positive = [float(value) for value in values if float(value) > 0.0]
    if not positive:
        return None
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def _group_geometric_means(
    cases: Sequence[dict[str, Any]],
    key_function: Any,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for case in cases:
        if case.get("converged") and case.get("r_c") is not None:
            grouped.setdefault(str(key_function(case)), []).append(float(case["r_c"]))
    return {
        key: float(_geometric_mean(grouped[key]))
        for key in sorted(grouped)
    }


def build_report_payload(results: dict[str, Any]) -> dict[str, Any]:
    result_cases = results.get("cases", {})
    ordered_cases = [result_cases[key] for key in sorted(result_cases)]
    converged = [
        case for case in ordered_cases
        if (
            not str(case.get("id", "")).startswith("C02-S")
            and case.get("converged")
            and case.get("r_c") is not None
            and float(case["r_c"]) > 0.0
        )
    ]
    ratios = [float(case["r_c"]) for case in converged]
    converged_strict = [
        case for case in ordered_cases
        if (
            not str(case.get("id", "")).startswith("C02-S")
            and case.get("converged_strict")
            and case.get("r_c_strict") is not None
            and float(case["r_c_strict"]) > 0.0
        )
    ]
    strict_ratios = [float(case["r_c_strict"]) for case in converged_strict]
    lenient_ratios = []
    for case in ordered_cases:
        if str(case.get("id", "")).startswith("C02-S"):
            continue
        prediction = case.get("requirement_prediction") or {}
        ceiling = prediction.get("ceiling_formation_g_cm2")
        a_conv = case.get("a_conv_g_cm2")
        if a_conv is None:
            a_conv = case.get("a_conv_lenient_g_cm2")
        if (
            a_conv is not None
            and ceiling is not None
            and float(a_conv) > 0.0
            and float(ceiling) > 0.0
        ):
            lenient_ratios.append(float(a_conv) / float(ceiling))
    geomean = _geometric_mean(ratios)
    recorded_zones = []
    intensity_correction_values = set()
    for case in converged:
        prediction = case.get("requirement_prediction") or {}
        zones_value = prediction.get("zones_per_scale_length")
        if zones_value is not None:
            recorded_zones.append(float(zones_value))
        correction_value = prediction.get("intensity_correction_factor")
        if correction_value is not None:
            intensity_correction_values.add(float(correction_value))
    zones_recorded = (
        recorded_zones[0]
        if (
            recorded_zones
            and len(recorded_zones) == len(converged)
            and all(value == recorded_zones[0] for value in recorded_zones)
        )
        else None
    )
    intensity_correction_recorded = sorted(intensity_correction_values)
    calibration = {
        "n_converged_cases": len(converged),
        "r_c_geometric_mean": geomean,
        "r_c_lenient_geometric_mean": _geometric_mean(lenient_ratios),
        "r_c_strict_geometric_mean": _geometric_mean(strict_ratios),
        "r_c_min": min(ratios) if ratios else None,
        "r_c_max": max(ratios) if ratios else None,
        "zones_per_scale_length_recorded": zones_recorded,
        "zones_per_scale_length_mean_matched": (
            None
            if (geomean is None or zones_recorded is None)
            else zones_recorded / geomean
        ),
        "zones_per_scale_length_all_cases_safe": (
            None
            if (not ratios or zones_recorded is None)
            else zones_recorded / min(ratios)
        ),
        "intensity_correction_recorded": intensity_correction_recorded,
        "group_geometric_means": {
            "wavelength_nm": _group_geometric_means(
                converged, lambda case: case["wavelength_nm"]
            ),
            "waveform": _group_geometric_means(
                converged, lambda case: case["waveform"]
            ),
            "density": _group_geometric_means(
                converged,
                lambda case: "/".join(
                    "{0:g}".format(float(layer["rho"])) for layer in case["layers"]
                ),
            ),
        },
    }
    report_cases = []
    for case in ordered_cases:
        prediction = case.get("requirement_prediction") or {}
        report_cases.append(
            {
                "id": case["id"],
                "describe": case["describe"],
                "geometry": case["geometry"],
                "wavelength_nm": case["wavelength_nm"],
                "waveform": case["waveform"],
                "layers": case["layers"],
                "rho_c_gcc": prediction.get("rho_c_gcc"),
                "ceiling_formation_g_cm2": prediction.get("ceiling_formation_g_cm2"),
                "a_conv_g_cm2": case.get("a_conv_g_cm2"),
                "a_conv_nominal_g_cm2": case.get("a_conv_nominal_g_cm2"),
                "a_conv_lenient_g_cm2": case.get("a_conv_lenient_g_cm2"),
                "a_conv_lenient_nominal_g_cm2": case.get(
                    "a_conv_lenient_nominal_g_cm2"
                ),
                "a_conv_strict_g_cm2": case.get("a_conv_strict_g_cm2"),
                "a_conv_strict_nominal_g_cm2": case.get(
                    "a_conv_strict_nominal_g_cm2"
                ),
                "r_c": case.get("r_c"),
                "r_c_strict": case.get("r_c_strict"),
                "converged": bool(case.get("converged")),
                "converged_strict": bool(case.get("converged_strict")),
                "monotone": bool(case.get("monotone")),
                "verdict_observables": case.get("verdict_observables", []),
                "notes": case.get("notes", []),
                "deviations": case.get("deviations", {}),
                "reference_last_increment": case.get(
                    "reference_last_increment"
                ),
                "convergence_rate_estimate": case.get(
                    "convergence_rate_estimate"
                ),
                "n_cells_per_level": {
                    level: data.get("n_cells")
                    for level, data in sorted(case.get("level_data", {}).items())
                },
                "P_rear_max_dyn_cm2_per_level": {
                    level: data.get("P_rear_max_dyn_cm2")
                    for level, data in sorted(case.get("level_data", {}).items())
                },
            }
        )
    return {
        "schema": "tenryu.mesh_convergence_reference.v1",
        "generated": datetime.now(timezone.utc).isoformat(),
        "tolerances": TOLERANCES,
        "ladder": A_SURF_LADDER,
        "cases": report_cases,
        "calibration": calibration,
    }


def render_markdown(results: dict[str, Any], payload: dict[str, Any]) -> str:
    calibration = payload["calibration"]
    zones_recorded = calibration["zones_per_scale_length_recorded"]
    zones = (
        "not recorded" if zones_recorded is None else _format_float(zones_recorded)
    )
    intensity_corrections = calibration["intensity_correction_recorded"]
    correction = (
        ", ".join(_format_float(value) for value in intensity_corrections)
        if intensity_corrections
        else "none / not recorded"
    )
    rows = [
        "# TENRYU mesh-convergence reference",
        "",
        (
            "Empirical grid-convergence reference for the 1D laser-ablation "
            "mesh generator (campaign design: "
            "docs/design/mesh_convergence_campaign_20260903.md). Each case "
            "was run over a ladder of surface areal-mass zonings; `a_conv` "
            "is the coarsest surface cell areal mass [g/cm^2] for which every "
            "finer level stays within the tolerances (P_a 5 %, m_abl 10 %, "
            "E_abs 3 %, t_bo 2 %; rho_R 10 % in the strict variant) of the "
            "finest level. `Predicted ceiling` is the formation-band ceiling "
            "of the a-priori resolution-requirement model AS CONFIGURED IN "
            "THE RUNS (zones_per_scale_length {zones}; intensity correction "
            "{corr}); `r_c` = a_conv / predicted ceiling. Calibration keys in "
            "the JSON payload are relative to those recorded parameters; the "
            "adopted defaults are documented in "
            "docs/design/mesh_resolution_requirement_20260903.md §7.3."
        ).format(zones=zones, corr=correction),
        "",
        "| ID | Description | lambda [nm] | Waveform | rho0 profile | verdict obs. | rho_c [g/cc] | Predicted ceiling [g/cm^2] | a_conv nominal [g/cm^2] | a_conv achieved [g/cm^2] | a_conv lenient | a_conv strict (with rho_R) | r_c | r_c strict | last increment P_a | rate | Levels run | Converged? | Notes |",
        "|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    result_cases = results.get("cases", {})
    for case_id in sorted(result_cases):
        if case_id.startswith("C02-S"):
            continue
        case = result_cases[case_id]
        prediction = case.get("requirement_prediction") or {}
        missing = sorted(
            name
            for deviations in case.get("deviations", {}).values()
            for name, value in deviations.items()
            if value is None
        )
        notes = list(case.get("notes", []))
        if missing:
            notes.append("missing: " + ", ".join(sorted(set(missing))))
        if not case.get("converged"):
            notes.append("reference not converged")
        last_increment = case.get("reference_last_increment") or {}
        rows.append(
            "| {id} | {describe} | {wavelength} | {waveform} | {profile} | {verdict_observables} | {rho_c} | {ceiling} | {a_conv_nominal} | {a_conv} | {a_conv_lenient} | {a_conv_strict} | {r_c} | {r_c_strict} | {last_increment_P_a} | {rate} | {levels} | {converged} | {notes} |".format(
                id=case_id,
                describe=str(case["describe"]).replace("|", "\\|"),
                wavelength=case["wavelength_nm"],
                waveform=case["waveform"],
                profile=_profile_label(case["layers"]).replace("|", "\\|"),
                verdict_observables=",".join(case.get("verdict_observables", [])),
                rho_c=_format_float(prediction.get("rho_c_gcc")),
                ceiling=_format_float(prediction.get("ceiling_formation_g_cm2")),
                a_conv_nominal=_format_float(case.get("a_conv_nominal_g_cm2")),
                a_conv=_format_float(case.get("a_conv_g_cm2")),
                a_conv_lenient=_format_float(case.get("a_conv_lenient_g_cm2")),
                a_conv_strict=_format_float(case.get("a_conv_strict_g_cm2")),
                r_c=_format_float(case.get("r_c")),
                r_c_strict=_format_float(case.get("r_c_strict")),
                last_increment_P_a=_format_float(last_increment.get("P_a")),
                rate=_format_float(case.get("convergence_rate_estimate")),
                levels=", ".join(str(level) for level in case.get("levels_run", [])),
                converged="yes" if case.get("converged") else "no",
                notes="; ".join(notes),
            )
        )
    sanity_cases = [
        result_cases[case_id]
        for case_id in sorted(result_cases)
        if case_id.startswith("C02-S")
    ]
    if sanity_cases:
        rows.extend(
            [
                "",
                "## Sanity studies",
                "",
                "| ID | Level | Varied quantity | Value | P_a deviation | m_abl deviation | E_abs deviation | rho_R deviation | t_bo deviation |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for case in sanity_cases:
            ladder_kind = case.get("ladder_kind")
            value_key = {
                "interior": "interior_areal_mass_g_cm2",
                "cfl": "cfl_hydro",
            }.get(ladder_kind)
            for level in case.get("levels_run", []):
                data = case.get("level_data", {}).get(str(level), {})
                deviations = case.get("deviations", {}).get(str(level), {})
                rows.append(
                    "| {id} | {level} | {quantity} | {value} | {P_a} | {m_abl} | {E_abs} | {rho_R} | {t_bo} |".format(
                        id=case["id"],
                        level=level,
                        quantity=value_key or "—",
                        value=_format_float(data.get(value_key)) if value_key else "—",
                        P_a=_format_float(deviations.get("P_a")),
                        m_abl=_format_float(deviations.get("m_abl")),
                        E_abs=_format_float(deviations.get("E_abs")),
                        rho_R=_format_float(deviations.get("rho_R")),
                        t_bo=_format_float(deviations.get("t_bo")),
                    )
                )
    rows.extend(
        [
            "",
            "## Calibration",
            "",
            "- Converged cases: {0}".format(calibration["n_converged_cases"]),
            "- Geometric mean r_c: {0}".format(_format_float(calibration["r_c_geometric_mean"])),
            "- Geometric mean r_c lenient: {0}".format(
                _format_float(calibration["r_c_lenient_geometric_mean"])
            ),
            "- Geometric mean r_c strict (with rho_R): {0}".format(
                _format_float(calibration["r_c_strict_geometric_mean"])
            ),
            "- r_c spread (min/max): {0} / {1}".format(
                _format_float(calibration["r_c_min"]),
                _format_float(calibration["r_c_max"]),
            ),
            "- Recorded zones_per_scale_length: {0}".format(
                _format_float(calibration["zones_per_scale_length_recorded"])
            ),
            "- Mean-matched zones_per_scale_length: {0}".format(
                _format_float(
                    calibration["zones_per_scale_length_mean_matched"]
                )
            ),
            "- All-cases-safe zones_per_scale_length: {0}".format(
                _format_float(
                    calibration["zones_per_scale_length_all_cases_safe"]
                )
            ),
        ]
    )
    for group_name, means in calibration["group_geometric_means"].items():
        rows.extend(["", "### By {0}".format(group_name), ""])
        if means:
            rows.extend(
                "- {0}: {1}".format(key, _format_float(value))
                for key, value in means.items()
            )
        else:
            rows.append("- No converged cases")
    return "\n".join(rows) + "\n"


def write_report(
    results_path: Path,
    markdown_path: Path,
    json_path: Path,
) -> dict[str, Any]:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if results.get("schema") != "tenryu.mesh_convergence.v1":
        raise CampaignError("unsupported results schema")
    payload = build_report_payload(results)
    _write_text_if_changed(markdown_path, render_markdown(results, payload))
    _write_json(json_path, payload)
    return payload


def print_case_table(sanity: bool = False) -> None:
    print("ID  Geometry   lambda_nm  Waveform  Description")
    for case in CASES + (SANITY_CASES if sanity else []):
        print(
            "{0:<3} {1:<10} {2:>9}  {3:<8}  {4}".format(
                case["id"], case["geometry"], case["wavelength_nm"],
                case["waveform"], case["describe"]
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("gen", help="generate campaign decks")
    generate.add_argument("--root", required=True)
    generate.add_argument("--cases")
    generate.add_argument(
        "--levels",
        help="comma-separated levels/ranges (default: all valid levels, 0-8 for campaign cases)",
    )
    generate.add_argument("--sanity", action="store_true")

    run = subparsers.add_parser("run", help="run campaign decks sequentially")
    run.add_argument("--root", required=True)
    run.add_argument("--tenryu", required=True)
    run.add_argument("--cases")
    run.add_argument("--adaptive", action="store_true")
    run.add_argument("--max-level", type=int)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")

    analyze = subparsers.add_parser("analyze", help="analyze campaign outputs")
    analyze.add_argument("--root", required=True)
    analyze.add_argument("--cases")
    analyze.add_argument("-o", "--output", default="results.json")

    report = subparsers.add_parser("report", help="write reference reports")
    report.add_argument("--root", required=True)
    report.add_argument("--results", required=True)
    report.add_argument("--md", required=True)
    report.add_argument("--json", required=True)

    list_cases = subparsers.add_parser("list", help="print the campaign case table")
    list_cases.add_argument("--sanity", action="store_true")
    return parser


def _rooted(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    try:
        if args.command == "list":
            print_case_table(args.sanity)
            return 0
        root = Path(args.root)
        if args.command == "gen":
            cases = select_cases(args.cases)
            paths = []
            for case in cases:
                paths.extend(
                    generate_decks(
                        root, [case], parse_levels(args.levels, case), False
                    )
                )
            if args.sanity:
                paths.extend(generate_decks(root, [], [], True))
            print("generated {0} decks".format(len(paths)))
            return 0
        if args.command == "run":
            return run_campaign(
                root,
                Path(args.tenryu),
                select_cases(args.cases),
                args.adaptive,
                args.max_level,
                args.resume,
                args.dry_run,
            )
        if args.command == "analyze":
            payload = analyze_campaign(root, select_cases(args.cases))
            output = _rooted(root, args.output)
            _write_json(output, payload)
            print(str(output))
            return 0
        if args.command == "report":
            write_report(
                _rooted(root, args.results),
                _rooted(root, args.md),
                _rooted(root, args.json),
            )
            return 0
    except (CampaignError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print("mesh_convergence_campaign: {0}".format(exc), file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
