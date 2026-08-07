#!/usr/bin/env python3
"""Compose declarative geometry primitives into TENRYU Mesh kwargs."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math


PI = 3.141592653589793
CENTER_BUTTON_A_MAX = 1.5
CENTER_BUTTON_BRIDGE_MARGIN = 1.15


# ---------- primitives (all lengths cm, densities g/cc, angles radian) ----


@dataclass
class PolarShell:
    r_in: float
    r_out: float
    rho: float
    material: str = ""


@dataclass
class PolarSolidSphere:
    r: float
    rho: float
    material: str = ""


@dataclass
class SlabLayer:
    z_from: float
    z_to: float
    rho: float
    material: str = ""


@dataclass
class RectRadialLayer:
    """Hohlraum barrel wall or other rectangular-radial layer."""

    r_from: float
    r_to: float
    rho: float
    material: str = ""


@dataclass
class Cone:
    """Apex-on-axis cone attached to a sphere or shell."""

    theta_half: float
    n_wall: int = 8
    refine_band: float = 0.05
    exterior_only: bool = False


@dataclass
class StalkWedge:
    """Thin theta-wedge stalk approximation."""

    theta_from: float
    theta_to: float
    n: int = 4


@dataclass
class VoidFill:
    rho: float = 1.0e-6


# ---------- base topologies --------------------------------------------


@dataclass
class PolarBase:
    s_max: float
    center_treatment: str = "tri_fan"
    s_inner: float = 0.0
    multiblock_r_match: Optional[float] = None
    n_theta_default: int = 32
    center_mode: Optional[str] = None
    center_seam_radius: Optional[float] = None
    center_core_ratio: Optional[float] = None
    center_bridge_layers: Optional[int] = None


@dataclass
class RectBase:
    r_min: float
    r_max: float
    z_min: float
    z_max: float


@dataclass
class Quality:
    mass_ratio_max: float = 1.3
    n_radial_hint: int = 0
    n_z_hint: int = 0
    default_zones_per_region: int = 16


@dataclass
class SmoothZoning:
    """Configure smooth zoning.

    ``interface_refine=1.0`` imposes no material-interface end-thickness
    constraint.  Values greater than 1.0 opt into the former shared interface
    end thickness with refinement.
    """

    ratio_max: float = 1.15
    feather_side: str = "outer"
    feather_factor: float = 20.0
    feather_ratio: float = 1.12
    n_total: int = 0
    min_zones_per_region: int = 4
    interface_refine: float = 1.0
    feather_interfaces: List[float] = field(default_factory=list)
    feather_factor_interface: float = 10.0
    feather_ratio_interface: float = 1.12
    feather_hold_depth: float = 0.0
    feather_hold_depth_interface: float = 0.0


@dataclass
class MeshPlan:
    mesh_kwargs: Dict
    report: Dict


@dataclass
class _MaterialInterval:
    start: float
    end: float
    rho: float
    material: str
    source: str = field(repr=False)


@dataclass
class _SmoothRegion:
    interval: _MaterialInterval
    is_void: bool
    n: int
    I_raw: float
    natural: float
    masses_plus: List[float]
    nodes: List[float]
    is_feather: bool = False


def _tolerance(value: float) -> float:
    return 1.0e-12 * max(1.0, abs(value))


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= 1.0e-12 * max(
        1.0, abs(left), abs(right)
    )


def _dedupe(values: List[float]) -> List[float]:
    result: List[float] = []
    for value in sorted(float(item) for item in values):
        if not result or not _close(value, result[-1]):
            result.append(value)
    return result


def _check_overlaps(intervals: List[_MaterialInterval]) -> None:
    ordered = sorted(intervals, key=lambda item: (item.start, item.end))
    for previous, current in zip(ordered, ordered[1:]):
        if current.start < previous.end - _tolerance(previous.end):
            raise ValueError(
                "overlapping material intervals: "
                f"{previous.source} and {current.source}"
            )


def _compose_intervals(
    materials: List[_MaterialInterval],
    domain_start: float,
    domain_end: float,
    void_rho: float,
) -> List[_MaterialInterval]:
    _check_overlaps(materials)
    result: List[_MaterialInterval] = []
    cursor = float(domain_start)

    for material in sorted(materials, key=lambda item: (item.start, item.end)):
        start = max(float(material.start), float(domain_start))
        end = min(float(material.end), float(domain_end))
        if end <= domain_start or start >= domain_end or _close(start, end):
            continue
        if _close(start, cursor):
            start = cursor
        if start > cursor + _tolerance(cursor):
            result.append(
                _MaterialInterval(cursor, start, float(void_rho), "", "VoidFill")
            )
        if end > cursor:
            result.append(
                _MaterialInterval(
                    max(start, cursor),
                    end,
                    float(material.rho),
                    str(material.material),
                    material.source,
                )
            )
            cursor = end

    if cursor < domain_end - _tolerance(domain_end):
        result.append(
            _MaterialInterval(
                cursor, float(domain_end), float(void_rho), "", "VoidFill"
            )
        )
    elif result:
        result[-1].end = float(domain_end)

    return result


def _allocate_counts(
    intervals: List[_MaterialInterval],
    hint: int,
    default_count: int,
    geometry_power: int,
) -> List[int]:
    if hint <= 0:
        return [int(default_count) for _ in intervals]

    measures = [
        float(interval.rho)
        * (float(interval.end) ** geometry_power
           - float(interval.start) ** geometry_power)
        for interval in intervals
    ]
    total_measure = sum(measures)
    if total_measure <= 0.0:
        raise ValueError("total interval mass measure must be positive")

    counts = [
        max(2, int(round(int(hint) * measure / total_measure)))
        for measure in measures
    ]
    difference = int(hint) - sum(counts)
    order = sorted(range(len(measures)), key=lambda index: measures[index], reverse=True)
    if difference > 0:
        counts[order[0]] += difference
    elif difference < 0:
        remaining = -difference
        for index in order:
            removable = max(0, counts[index] - 2)
            removed = min(removable, remaining)
            counts[index] -= removed
            remaining -= removed
            if remaining == 0:
                break

    return [int(count) for count in counts]


def _boundaries(intervals: List[_MaterialInterval]) -> List[float]:
    if not intervals:
        return []
    return [float(intervals[0].start)] + [
        float(interval.end) for interval in intervals
    ]


def _auto_regions(
    intervals: List[_MaterialInterval], counts: List[int], void_rho: float
) -> List[Dict]:
    return [
        {
            "r_end": float(interval.end),
            "nz": int(count),
            "rho_ref": float(interval.rho),
            "is_void": bool(interval.rho <= 10.0 * void_rho),
            "material_group": str(interval.material),
        }
        for interval, count in zip(intervals, counts)
    ]


def _segments(
    intervals: List[_MaterialInterval], counts: List[int]
) -> List[Dict]:
    return [
        {
            "r_start": float(interval.start),
            "r_end": float(interval.end),
            "nr": int(count),
        }
        for interval, count in zip(intervals, counts)
    ]


def _record_direction(
    report: Dict,
    direction: str,
    interfaces: List[float],
    counts: List[int],
    notes: Optional[List[str]] = None,
) -> None:
    report["directions"][direction] = {
        "interfaces": [float(value) for value in interfaces],
        "zone_counts": [int(count) for count in counts],
        "notes": [str(note) for note in (notes or [])],
    }
    report["interfaces_pinned"].extend(
        (str(direction), float(value)) for value in interfaces
    )


def _finish_total_cells(report: Dict) -> None:
    totals = [
        sum(direction["zone_counts"])
        for direction in report["directions"].values()
    ]
    if not totals:
        report["total_cells"] = None
    elif len(totals) == 1:
        report["total_cells"] = int(totals[0])
    else:
        report["total_cells"] = int(math.prod(totals))


def _void_density(primitives: List) -> float:
    for primitive in primitives:
        if isinstance(primitive, VoidFill):
            return float(primitive.rho)
    return 1.0e-6


def _smooth_measure(value: float, geometry_power: int) -> float:
    if geometry_power == 3:
        return float(value) ** 3
    return float(value)


def _smooth_inverse_measure(value: float, geometry_power: int) -> float:
    if geometry_power == 3:
        return float(value) ** (1.0 / 3.0)
    return float(value)


def _smooth_ramp_integral(
    length: float,
    end_size: float,
    slope: float,
    natural: float,
    hold: float = 0.0,
) -> float:
    if float(hold) != 0.0:
        if slope == 0.0:
            return float(length) / min(float(end_size), float(natural))
        hold_length = min(float(length), float(hold))
        hold_integral = hold_length / min(
            float(end_size), float(natural)
        )
        return hold_integral + _smooth_ramp_integral(
            float(length) - hold_length,
            end_size,
            slope,
            natural,
        )
    if slope == 0.0:
        return float(length) / min(float(end_size), float(natural))

    cap_distance = max(
        0.0,
        (float(natural) - float(end_size)) / float(slope),
    )
    ramp_length = min(float(length), cap_distance)
    integral = math.log1p(
        float(slope) * ramp_length / float(end_size)
    ) / float(slope)
    return integral + (float(length) - ramp_length) / float(natural)


def _smooth_ramp_inverse(
    cumulative: float,
    end_size: float,
    slope: float,
    natural: float,
    hold: float = 0.0,
) -> float:
    if float(hold) != 0.0:
        if slope == 0.0:
            return float(cumulative) * min(
                float(end_size), float(natural)
            )
        hold_integral = float(hold) / min(
            float(end_size), float(natural)
        )
        if float(cumulative) <= hold_integral:
            return float(cumulative) * min(
                float(end_size), float(natural)
            )
        return float(hold) + _smooth_ramp_inverse(
            float(cumulative) - hold_integral,
            end_size,
            slope,
            natural,
        )
    if slope == 0.0:
        return float(cumulative) * min(float(end_size), float(natural))

    cap_distance = max(
        0.0,
        (float(natural) - float(end_size)) / float(slope),
    )
    cap_integral = math.log1p(
        float(slope) * cap_distance / float(end_size)
    ) / float(slope)
    if float(cumulative) <= cap_integral:
        return float(end_size) * math.expm1(
            float(cumulative) * float(slope)
        ) / float(slope)
    return cap_distance + (
        float(cumulative) - cap_integral
    ) * float(natural)


def _smooth_crossover(
    length: float,
    left_target: float,
    right_target: float,
    slope: float,
    left_hold: float = 0.0,
    right_hold: float = 0.0,
) -> float:
    if float(left_hold) != 0.0 or float(right_hold) != 0.0:
        def difference(distance: float) -> float:
            left_size = float(left_target) + float(slope) * max(
                0.0, float(distance) - float(left_hold)
            )
            right_size = float(right_target) + float(slope) * max(
                0.0,
                float(length) - float(distance) - float(right_hold),
            )
            return left_size - right_size

        if difference(0.0) >= 0.0:
            return 0.0
        if difference(float(length)) <= 0.0:
            return float(length)
        lower = 0.0
        upper = float(length)
        for _ in range(200):
            crossover = 0.5 * (lower + upper)
            delta = difference(crossover)
            if delta == 0.0:
                return crossover
            if delta < 0.0:
                lower = crossover
            else:
                upper = crossover
        return 0.5 * (lower + upper)
    crossover = (
        float(right_target) + float(slope) * float(length)
        - float(left_target)
    ) / (2.0 * float(slope))
    return min(float(length), max(0.0, crossover))


def _smooth_sizing_integral(
    length: float,
    natural: float,
    left_target: Optional[float],
    right_target: Optional[float],
    slope: float,
    left_hold: float = 0.0,
    right_hold: float = 0.0,
) -> float:
    if left_target is None and right_target is None:
        return float(length) / float(natural)
    if left_target is not None and right_target is None:
        return _smooth_ramp_integral(
            length, left_target, slope, natural, left_hold
        )
    if left_target is None and right_target is not None:
        return _smooth_ramp_integral(
            length, right_target, slope, natural, right_hold
        )
    if slope == 0.0:
        return float(length) / min(float(left_target), float(right_target))

    crossover = _smooth_crossover(
        length,
        float(left_target),
        float(right_target),
        slope,
        left_hold,
        right_hold,
    )
    return (
        _smooth_ramp_integral(
            crossover, float(left_target), slope, natural, left_hold
        )
        + _smooth_ramp_integral(
            float(length) - crossover,
            float(right_target),
            slope,
            natural,
            right_hold,
        )
    )


def _smooth_sizing_nodes(
    interval: _MaterialInterval,
    count: int,
    natural: float,
    left_target: Optional[float],
    right_target: Optional[float],
    slope: float,
    left_hold: float = 0.0,
    right_hold: float = 0.0,
) -> List[float]:
    start = float(interval.start)
    end = float(interval.end)
    length = end - start

    if left_target is None and right_target is None:
        nodes = [
            start + length * index / int(count)
            for index in range(int(count) + 1)
        ]
    elif left_target is not None and right_target is None:
        nodes = [start]
        for index in range(1, int(count)):
            distance = _smooth_ramp_inverse(
                index,
                float(left_target),
                float(slope),
                natural,
                left_hold,
            )
            nodes.append(start + distance)
        nodes.append(end)
    elif left_target is None and right_target is not None:
        nodes = [start]
        for index in range(1, int(count)):
            distance = _smooth_ramp_inverse(
                int(count) - index,
                float(right_target),
                float(slope),
                natural,
                right_hold,
            )
            nodes.append(end - distance)
        nodes.append(end)
    elif slope == 0.0:
        nodes = [
            start + length * index / int(count)
            for index in range(int(count) + 1)
        ]
    else:
        crossover = _smooth_crossover(
            length,
            float(left_target),
            float(right_target),
            slope,
            left_hold,
            right_hold,
        )
        crossover_count = _smooth_ramp_integral(
            crossover,
            float(left_target),
            float(slope),
            natural,
            left_hold,
        )
        nodes = [start]
        for index in range(1, int(count)):
            if index <= crossover_count:
                distance = _smooth_ramp_inverse(
                    index,
                    float(left_target),
                    float(slope),
                    natural,
                    left_hold,
                )
                nodes.append(start + distance)
            else:
                distance = _smooth_ramp_inverse(
                    int(count) - index,
                    float(right_target),
                    float(slope),
                    natural,
                    right_hold,
                )
                nodes.append(end - distance)
        nodes.append(end)

    nodes[0] = start
    nodes[-1] = end
    return nodes


def _smooth_region(
    interval: _MaterialInterval,
    is_void: bool,
    n_void: int,
    min_count: int,
    natural: float,
    left_target: Optional[float],
    right_target: Optional[float],
    left_hold: float,
    right_hold: float,
    slope: float,
    geometry_power: int,
    is_feather: bool,
    warnings: List[str],
) -> _SmoothRegion:
    length = float(interval.end) - float(interval.start)
    if is_void:
        I_raw = float(n_void)
        count = int(n_void)
        nodes = [
            float(interval.start) + length * index / count
            for index in range(count + 1)
        ]
        nodes[0] = float(interval.start)
        nodes[-1] = float(interval.end)
    else:
        I_raw = _smooth_sizing_integral(
            length,
            natural,
            left_target,
            right_target,
            slope,
            left_hold,
            right_hold,
        )
        constrained_targets = [
            float(target)
            for target in (left_target, right_target)
            if target is not None
        ]
        node_slope = float(slope)
        if not constrained_targets:
            count = max(int(min_count), round(I_raw))
        else:
            count = max(int(min_count), math.ceil(I_raw))
            minimum_target = min(constrained_targets)
            I_limit = length / minimum_target
            if count <= math.floor(I_limit):
                relative_error = abs(I_raw - count) / count
                if relative_error > 1.0e-13:
                    lower = 0.0
                    upper = float(slope)
                    for _ in range(200):
                        node_slope = 0.5 * (lower + upper)
                        integral = _smooth_sizing_integral(
                            length,
                            natural,
                            left_target,
                            right_target,
                            node_slope,
                            left_hold,
                            right_hold,
                        )
                        if abs(integral - count) / count <= 1.0e-13:
                            break
                        if integral > count:
                            lower = node_slope
                        else:
                            upper = node_slope
                    else:
                        raise RuntimeError(
                            "smooth zoning slope solve failed: "
                            f"region={interval.material!r}, "
                            f"I_raw={I_raw}, I_limit={I_limit}, N={count}"
                        )
            else:
                beta = (length / count) / minimum_target
                if left_target is not None:
                    left_target = beta * float(left_target)
                if right_target is not None:
                    right_target = beta * float(right_target)
                node_slope = 0.0
                warnings.append(
                    "smooth zoning thin-region degradation: "
                    f"region={interval.material!r} "
                    f"[{interval.start}, {interval.end}], beta={beta}"
                )
        nodes = _smooth_sizing_nodes(
            interval,
            count,
            natural,
            left_target,
            right_target,
            node_slope,
            left_hold,
            right_hold,
        )

    masses_plus = [
        float(interval.rho) * (
            _smooth_measure(right, geometry_power)
            - _smooth_measure(left, geometry_power)
        )
        for left, right in zip(nodes, nodes[1:])
    ]
    return _SmoothRegion(
        interval=interval,
        is_void=bool(is_void),
        n=int(count),
        I_raw=float(I_raw),
        natural=float(natural),
        masses_plus=masses_plus,
        nodes=nodes,
        is_feather=bool(is_feather),
    )


def _smooth_feather_target(
    interval: _MaterialInterval,
    n_default: int,
    feather_factor: float,
    geometry_power: int,
    feather_side: str,
) -> float:
    measure_length = (
        _smooth_measure(interval.end, geometry_power)
        - _smooth_measure(interval.start, geometry_power)
    )
    measure_cell = measure_length / int(n_default) / float(feather_factor)
    if geometry_power == 3:
        if feather_side == "outer":
            surface = float(interval.end)
            target = surface - _smooth_inverse_measure(
                _smooth_measure(surface, geometry_power) - measure_cell,
                geometry_power,
            )
        else:
            surface = float(interval.start)
            target = _smooth_inverse_measure(
                _smooth_measure(surface, geometry_power) + measure_cell,
                geometry_power,
            ) - surface
    else:
        target = measure_cell
    natural = (
        float(interval.end) - float(interval.start)
    ) / int(n_default)
    return min(float(target), natural)


def _smooth_feather_count(
    records: List[_SmoothRegion], feather_side: str
) -> int:
    if feather_side == "none":
        return 0
    record = records[0] if feather_side == "inner" else records[-1]
    if record.is_void or not record.is_feather:
        return 0
    widths = [
        float(right) - float(left)
        for left, right in zip(record.nodes, record.nodes[1:])
    ]
    if feather_side == "outer":
        widths.reverse()
    for count, width in enumerate(widths, start=1):
        if abs(width - record.natural) <= 0.05 * record.natural:
            return int(count)
    return int(record.n)


def _smooth_plan_intervals(
    intervals: List[_MaterialInterval],
    void_rho: float,
    quality: Quality,
    smooth: SmoothZoning,
    geometry_power: int,
) -> Tuple[
    List[float], List[int], Dict, List[str], List[_SmoothRegion]
]:
    if float(smooth.feather_hold_depth) < 0.0:
        raise ValueError("feather_hold_depth must be >= 0")
    if float(smooth.feather_hold_depth_interface) < 0.0:
        raise ValueError("feather_hold_depth_interface must be >= 0")

    ratios = [float(smooth.ratio_max), float(smooth.feather_ratio)]
    if smooth.feather_interfaces:
        ratios.append(float(smooth.feather_ratio_interface))
    slope = max(ratios) - 1.0
    n_default = max(
        int(smooth.min_zones_per_region),
        int(quality.default_zones_per_region),
    )
    n_void = max(
        int(smooth.min_zones_per_region),
        int(quality.default_zones_per_region) // 2,
    )
    void_flags = [
        bool(float(interval.rho) <= 10.0 * float(void_rho))
        for interval in intervals
    ]
    naturals = [
        (float(interval.end) - float(interval.start)) / n_default
        for interval in intervals
    ]
    left_targets: List[Optional[float]] = [None for _ in intervals]
    right_targets: List[Optional[float]] = [None for _ in intervals]
    left_holds = [0.0 for _ in intervals]
    right_holds = [0.0 for _ in intervals]
    available_interfaces = [
        float(interval.end) for interval in intervals[:-1]
    ]
    feather_interface_indices: List[int] = []
    for requested in smooth.feather_interfaces:
        matched_index = next(
            (
                index
                for index, interface in enumerate(available_interfaces)
                if math.isclose(
                    float(requested), interface, rel_tol=1.0e-9
                )
            ),
            None,
        )
        if matched_index is None:
            available_text = ", ".join(
                f"{interface:.3f}" for interface in available_interfaces
            )
            raise ValueError(
                "smooth zoning feather interface "
                f"{float(requested)} is not a material interface; "
                f"available interfaces: [{available_text}]"
            )
        feather_interface_indices.append(matched_index)

    if float(smooth.interface_refine) > 1.0:
        for index in range(len(intervals) - 1):
            left_void = void_flags[index]
            right_void = void_flags[index + 1]
            if not left_void and not right_void:
                target = min(naturals[index], naturals[index + 1]) / float(
                    smooth.interface_refine
                )
                right_targets[index] = target
                left_targets[index + 1] = target
            elif left_void and not right_void:
                left_targets[index + 1] = naturals[index + 1] / float(
                    smooth.interface_refine
                )
            elif not left_void and right_void:
                right_targets[index] = naturals[index] / float(
                    smooth.interface_refine
                )

    for index in feather_interface_indices:
        left_interval = intervals[index]
        right_interval = intervals[index + 1]
        right_targets[index] = None
        left_targets[index + 1] = None
        if float(left_interval.rho) > float(right_interval.rho):
            left_length = (
                float(left_interval.end) - float(left_interval.start)
            )
            if float(smooth.feather_hold_depth_interface) >= left_length or (
                math.isclose(
                    float(smooth.feather_hold_depth_interface),
                    left_length,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                )
            ):
                raise ValueError(
                    "feather_hold_depth_interface must be less than the "
                    "owning region's length"
                )
            right_targets[index] = _smooth_feather_target(
                left_interval,
                n_default,
                smooth.feather_factor_interface,
                geometry_power,
                "outer",
            )
            right_holds[index] = float(
                smooth.feather_hold_depth_interface
            )
        elif float(right_interval.rho) > float(left_interval.rho):
            right_length = (
                float(right_interval.end) - float(right_interval.start)
            )
            if float(smooth.feather_hold_depth_interface) >= right_length or (
                math.isclose(
                    float(smooth.feather_hold_depth_interface),
                    right_length,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                )
            ):
                raise ValueError(
                    "feather_hold_depth_interface must be less than the "
                    "owning region's length"
                )
            left_targets[index + 1] = _smooth_feather_target(
                right_interval,
                n_default,
                smooth.feather_factor_interface,
                geometry_power,
                "inner",
            )
            left_holds[index + 1] = float(
                smooth.feather_hold_depth_interface
            )

    feather_index: Optional[int] = None
    if smooth.feather_side == "inner" and intervals and not void_flags[0]:
        feather_index = 0
        inner_length = float(intervals[0].end) - float(intervals[0].start)
        if float(smooth.feather_hold_depth) >= inner_length or math.isclose(
            float(smooth.feather_hold_depth),
            inner_length,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            raise ValueError(
                "feather_hold_depth must be less than the owning region's "
                "length"
            )
        left_targets[0] = _smooth_feather_target(
            intervals[0],
            n_default,
            smooth.feather_factor,
            geometry_power,
            "inner",
        )
        left_holds[0] = float(smooth.feather_hold_depth)
    elif smooth.feather_side == "outer" and intervals and not void_flags[-1]:
        feather_index = len(intervals) - 1
        outer_length = float(intervals[-1].end) - float(
            intervals[-1].start
        )
        if float(smooth.feather_hold_depth) >= outer_length or math.isclose(
            float(smooth.feather_hold_depth),
            outer_length,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            raise ValueError(
                "feather_hold_depth must be less than the owning region's "
                "length"
            )
        right_targets[-1] = _smooth_feather_target(
            intervals[-1],
            n_default,
            smooth.feather_factor,
            geometry_power,
            "outer",
        )
        right_holds[-1] = float(smooth.feather_hold_depth)

    notes: List[str] = []
    records = [
        _smooth_region(
            interval,
            void_flags[index],
            n_void,
            smooth.min_zones_per_region,
            naturals[index],
            left_targets[index],
            right_targets[index],
            left_holds[index],
            right_holds[index],
            slope,
            geometry_power,
            index == feather_index,
            notes,
        )
        for index, interval in enumerate(intervals)
    ]

    nodes = [float(records[0].nodes[0])]
    for record in records:
        nodes.extend(float(value) for value in record.nodes[1:])

    max_within = 1.0
    for record in records:
        if (
            record.is_void
            or (
                geometry_power == 3
                and float(record.interval.start) == 0.0
            )
        ):
            continue
        for left, right in zip(record.masses_plus, record.masses_plus[1:]):
            max_within = max(max_within, left / right, right / left)

    max_interface_thickness = 1.0
    interface_mass_ratios: List[float] = []
    for left, right in zip(records, records[1:]):
        if left.is_void or right.is_void:
            continue
        left_thickness = float(left.nodes[-1] - left.nodes[-2])
        right_thickness = float(right.nodes[1] - right.nodes[0])
        max_interface_thickness = max(
            max_interface_thickness,
            left_thickness / right_thickness,
            right_thickness / left_thickness,
        )
        interface_mass_ratios.append(
            float(right.masses_plus[0] / left.masses_plus[-1])
        )

    smooth_report = {
        "cell_mass": [
            float(mass)
            for record in records
            for mass in record.masses_plus
        ],
        "per_region": [
            {
                "material": str(record.interval.material),
                "n": int(record.n),
                "I_raw": float(record.I_raw),
            }
            for record in records
        ],
        "max_within_region_mass_ratio": float(max_within),
        "max_interface_thickness_ratio": float(max_interface_thickness),
        "interface_mass_ratios": interface_mass_ratios,
        "n_feather": _smooth_feather_count(records, smooth.feather_side),
    }
    if float(smooth.feather_hold_depth) != 0.0:
        smooth_report["feather_hold_depth"] = float(
            smooth.feather_hold_depth
        )
    if float(smooth.feather_hold_depth_interface) != 0.0:
        smooth_report["feather_hold_depth_interface"] = float(
            smooth.feather_hold_depth_interface
        )
    realized_total = sum(record.n for record in records)
    if int(smooth.n_total) > 0 and realized_total != int(smooth.n_total):
        notes.append(
            "smooth zoning n_total is advisory: "
            f"requested {int(smooth.n_total)}, realized {realized_total}"
        )
    return (
        nodes,
        [int(record.n) for record in records],
        smooth_report,
        notes,
        records,
    )


def _validate_primitives(primitives: List, base) -> None:
    polar_only = (PolarShell, PolarSolidSphere, Cone, StalkWedge)
    rect_only = (SlabLayer, RectRadialLayer)
    allowed = polar_only + rect_only + (VoidFill,)

    for primitive in primitives:
        if not isinstance(primitive, allowed):
            raise ValueError(
                f"unsupported primitive {type(primitive).__name__}"
            )
        if isinstance(base, PolarBase) and isinstance(primitive, rect_only):
            raise ValueError(
                f"{type(primitive).__name__} is not valid with PolarBase"
            )
        if isinstance(base, RectBase) and isinstance(primitive, polar_only):
            raise ValueError(
                f"{type(primitive).__name__} is not valid with RectBase"
            )


def _plan_theta(
    primitives: List, base: PolarBase, mesh_kwargs: Dict, report: Dict
) -> None:
    cones = [
        item
        for item in primitives
        if isinstance(item, Cone) and not item.exterior_only
    ]
    stalks = [item for item in primitives if isinstance(item, StalkWedge)]
    if not cones and not stalks:
        return

    values = [0.0, PI]
    cone_intervals: List[Tuple[float, float, int]] = []
    for cone in cones:
        lower = float(round(
            min(PI, max(0.0, float(cone.theta_half - cone.refine_band))), 15
        ))
        middle = float(round(
            min(PI, max(0.0, float(cone.theta_half))), 15
        ))
        upper = float(round(
            min(PI, max(0.0, float(cone.theta_half + cone.refine_band))), 15
        ))
        values.extend([lower, middle, upper])
        cone_intervals.extend(
            [(lower, middle, int(cone.n_wall)),
             (middle, upper, int(cone.n_wall))]
        )

    stalk_intervals: List[Tuple[float, float, int]] = []
    for stalk in stalks:
        start = float(stalk.theta_from)
        end = float(stalk.theta_to)
        values.extend([start, end])
        stalk_intervals.append((start, end, int(stalk.n)))

    interfaces = _dedupe(values)
    counts: List[int] = []
    for start, end in zip(interfaces, interfaces[1:]):
        count: Optional[int] = None
        for refined_start, refined_end, refined_count in cone_intervals:
            if _close(start, refined_start) and _close(end, refined_end):
                count = int(refined_count)
                break
        if count is None:
            for wedge_start, wedge_end, wedge_count in stalk_intervals:
                if _close(start, wedge_start) and _close(end, wedge_end):
                    count = int(wedge_count)
                    break
        if count is None:
            count = max(
                2,
                int(round(int(base.n_theta_default) * (end - start) / PI)),
            )
        counts.append(int(count))

    theta_segments = [
        {"r_start": float(start), "r_end": float(end), "nr": int(count)}
        for start, end, count in zip(interfaces, interfaces[1:], counts)
    ]
    mesh_kwargs["grid_theta"] = {
        "type": "graded",
        "segments": theta_segments,
    }
    _record_direction(report, "theta", interfaces, counts)


def _theta_ladder(
    base: PolarBase, mesh_kwargs: Dict
) -> Tuple[int, List[float]]:
    if "explicit_nodes_theta" in mesh_kwargs:
        theta_nodes = [
            float(value) for value in mesh_kwargs["explicit_nodes_theta"]
        ]
        spacings = [
            right - left
            for left, right in zip(theta_nodes, theta_nodes[1:])
        ]
    elif "grid_theta" in mesh_kwargs:
        spacings = []
        for segment in mesh_kwargs["grid_theta"]["segments"]:
            count = int(segment["nr"])
            spacing = (
                float(segment["r_end"]) - float(segment["r_start"])
            ) / count
            spacings.extend([spacing] * count)
    else:
        n_theta_value = mesh_kwargs.get(
            "n_theta", getattr(base, "n_theta_default", None)
        )
        if n_theta_value is None:
            raise ValueError(
                "center_mode='graded_button' requires n_theta"
            )
        n_theta = int(n_theta_value)
        spacings = [PI / n_theta] * n_theta if n_theta > 0 else []

    if not spacings or any(spacing <= 0.0 for spacing in spacings):
        raise ValueError(
            "center_mode='graded_button' requires a positive theta ladder"
        )
    return len(spacings), spacings


def _apply_graded_button(
    base: PolarBase,
    mesh_kwargs: Dict,
    report: Dict,
    nodes: List[float],
    innermost_region: _SmoothRegion,
) -> None:
    n_theta, theta_spacings = _theta_ladder(base, mesh_kwargs)
    n_c = n_theta // 4
    if n_theta % 4 != 0 or n_c < 4:
        raise ValueError(
            "center_mode='graded_button' requires n_theta divisible by 4 "
            "and n_theta // 4 >= 4"
        )
    if base.center_core_ratio is not None:
        core_ratio = float(base.center_core_ratio)
        if not 0.0 < core_ratio < 1.0 / math.sqrt(2.0):
            raise ValueError(
                "center_core_ratio must be > 0 and < 1/sqrt(2), the "
                "sqrt(2) geometric bound"
            )
    if (
        base.center_bridge_layers is not None
        and int(base.center_bridge_layers) < 1
    ):
        raise ValueError("center_bridge_layers must be >= 1")

    h_s = float(innermost_region.natural)
    h_s_method = "smooth_region_natural"
    dtheta_min = min(theta_spacings)
    dtheta_max = max(theta_spacings)
    buffer = 4.0 * h_s
    innermost_outer = float(innermost_region.interval.end)
    if base.center_seam_radius is not None:
        requested_seam = float(base.center_seam_radius)
        candidate_indices = range(1, len(nodes) - 1)
        if not candidate_indices:
            raise ValueError(
                "center_seam_radius override requires an interior radial face"
            )
        k_sel = min(
            candidate_indices,
            key=lambda index: abs(float(nodes[index]) - requested_seam),
        )
        s_k = float(nodes[k_sel])
        h_r = float(nodes[k_sel + 1]) - s_k
        a_seam = max(
            h_r / (s_k * dtheta_min),
            (s_k * dtheta_max) / h_r,
        )
        failures = []
        if a_seam > CENTER_BUTTON_A_MAX:
            failures.append("A_k <= CENTER_BUTTON_A_MAX (a)")
        if min(h_r, s_k * dtheta_min) < (
            h_s / CENTER_BUTTON_A_MAX
        ):
            failures.append("spacing floor (b)")
        if s_k + buffer > innermost_outer:
            failures.append("material buffer (c)")
        if failures:
            raise ValueError(
                "center_seam_radius override failed condition(s): "
                + ", ".join(failures)
            )
        rejected_count = 0
        seam_policy = "override_nearest"
    else:
        failure_order = (
            "isotropy window (a)",
            "minimum spacing (b)",
            "material homogeneity (c)",
            "interior face (d)",
        )
        failure_counts = {failure: 0 for failure in failure_order}
        best_a = math.inf
        best_k: Optional[int] = None
        valid_candidates: List[Tuple[int, float]] = []
        rejected_count = 0

        for k in range(1, len(nodes) - 1):
            s_k = float(nodes[k])
            h_r = float(nodes[k + 1]) - s_k
            a_k = max(
                h_r / (s_k * dtheta_min),
                (s_k * dtheta_max) / h_r,
            )
            if a_k < best_a:
                best_a = a_k
                best_k = k

            failures = []
            if a_k > CENTER_BUTTON_A_MAX:
                failures.append("isotropy window (a)")
            if min(h_r, s_k * dtheta_min) < (
                h_s / CENTER_BUTTON_A_MAX
            ):
                failures.append("minimum spacing (b)")
            if s_k + buffer > innermost_outer:
                failures.append("material homogeneity (c)")
            if s_k >= float(base.s_max):
                failures.append("interior face (d)")

            if not failures:
                valid_candidates.append((k, a_k))
                continue
            rejected_count += 1
            for failure in failures:
                failure_counts[failure] += 1

        if not valid_candidates:
            most_frequent_failure = max(
                failure_order, key=lambda failure: failure_counts[failure]
            )
            raise ValueError(
                "no radial face satisfies the graded-button isotropy window; "
                f"best A_k={best_a} at k={best_k}; most frequently failed "
                f"constraint: {most_frequent_failure} "
                f"({failure_counts[most_frequent_failure]} candidates); "
                "consider increasing n_theta or the central region resolution"
            )

        s_full = (
            CENTER_BUTTON_BRIDGE_MARGIN * math.sqrt(2.0) * n_c * h_s
        )
        full_coarse_candidates = [
            candidate
            for candidate in valid_candidates
            if nodes[candidate[0]] >= s_full
        ]
        if full_coarse_candidates:
            k_sel, a_seam = full_coarse_candidates[0]
            seam_policy = "full_coarse_min"
        else:
            k_sel, a_seam = valid_candidates[-1]
            seam_policy = "partial_largest"
    s_b = nodes[k_sel]
    r_c_cap_geometry = s_b / (
        CENTER_BUTTON_BRIDGE_MARGIN * math.sqrt(2.0)
    )
    r_c_cap_scale = n_c * h_s
    if base.center_core_ratio is None:
        r_c = min(r_c_cap_geometry, r_c_cap_scale)
        r_c_binding_cap = (
            "geometry" if r_c_cap_geometry <= r_c_cap_scale else "scale"
        )
    else:
        r_c = float(base.center_core_ratio) * s_b
        r_c_binding_cap = "override"
    h_core = r_c / n_c
    if base.center_bridge_layers is None:
        # aligned-bridge uniformity law — the Shirley-Chiu morph maps the
        # bridge to the annulus [cbrt(1.5)*r_c, r_match]; N_b makes the
        # aligned rows uniform at h_s, eliminating the seam-crossing entropy
        # stripe (measured 6.82% -> 0.34%, 2026-07-21).
        bridge_layers = min(
            64,
            max(
                4,
                round(
                    (s_b - math.cbrt(1.5) * r_c) / h_s
                ),
            ),
        )
    else:
        bridge_layers = int(base.center_bridge_layers)
    shell_nodes = [s_b] + [value for value in nodes if value > s_b]

    mesh_kwargs["topology_scheme"] = (
        "multiblock_cart_core_polar_shell"
    )
    mesh_kwargs["multiblock_cart_core_r_match"] = s_b
    mesh_kwargs["multiblock_cart_core_r_c"] = r_c
    mesh_kwargs["multiblock_cart_core_n_c"] = n_c
    mesh_kwargs["multiblock_cart_core_bridge_layers"] = bridge_layers
    mesh_kwargs["multiblock_cart_core_bridge_grading"] = "quintic_log"
    mesh_kwargs["explicit_nodes"] = shell_nodes
    mesh_kwargs["nr"] = len(shell_nodes) - 1
    mesh_kwargs.pop("polar_center_treatment", None)

    report["center_button"] = {
        "seam_radius": s_b,
        "seam_face_index": k_sel,
        "seam_policy": seam_policy,
        "seam_source": (
            "auto" if base.center_seam_radius is None else "override"
        ),
        "A_seam": a_seam,
        "h_s": h_s,
        "h_s_method": h_s_method,
        "h_core": h_core,
        "h_core_over_h_s": h_core / h_s,
        "r_c_binding_cap": r_c_binding_cap,
        "core_ratio_source": (
            "auto" if base.center_core_ratio is None else "override"
        ),
        "n_c": n_c,
        "bridge_layers": bridge_layers,
        "bridge_layers_source": (
            "auto" if base.center_bridge_layers is None else "override"
        ),
        "bridge_grading": "quintic_log",
        "candidates_rejected": rejected_count,
        "cfl_gain_estimate": 1.0 / dtheta_max,
    }
    if base.center_bridge_layers is None:
        report["center_button"]["bridge_layers_law"] = (
            "aligned_uniform_h_s"
        )
    if base.center_seam_radius is not None:
        report["center_button"]["seam_radius_requested"] = float(
            base.center_seam_radius
        )
        report["center_button"]["seam_radius_snapped"] = s_b
    report["directions"]["r"]["notes"].append(
        "center_mode=graded_button: "
        f"seam at s_b={s_b} (face k={k_sel}, A={a_seam})"
    )


def _plan_polar(
    primitives: List,
    base: PolarBase,
    quality: Quality,
    smooth: Optional[SmoothZoning] = None,
) -> MeshPlan:
    exterior_cones = [
        primitive
        for primitive in primitives
        if isinstance(primitive, Cone) and primitive.exterior_only
    ]
    if len(exterior_cones) > 1:
        raise ValueError("at most one exterior_only Cone is supported")
    if base.center_mode not in (None, "graded_button"):
        raise ValueError("center_mode must be None or 'graded_button'")
    if base.center_mode == "graded_button":
        if smooth is None:
            raise ValueError(
                "center_mode='graded_button' requires SmoothZoning"
            )
        if base.center_treatment == "annular":
            raise ValueError(
                "center_mode='graded_button' is not valid with annular "
                "center treatment"
            )
        if base.multiblock_r_match is not None:
            raise ValueError(
                "center_mode='graded_button' is mutually exclusive with "
                "multiblock_r_match"
            )

    if base.center_treatment == "annular":
        for primitive in primitives:
            if isinstance(primitive, PolarSolidSphere):
                raise ValueError(
                    "PolarSolidSphere is not valid with annular center treatment"
                )

    void_rho = _void_density(primitives)
    materials: List[_MaterialInterval] = []
    primitive_interfaces: List[float] = []
    for primitive in primitives:
        if isinstance(primitive, PolarSolidSphere):
            materials.append(
                _MaterialInterval(
                    0.0,
                    float(primitive.r),
                    float(primitive.rho),
                    str(primitive.material),
                    "PolarSolidSphere",
                )
            )
            primitive_interfaces.extend([0.0, float(primitive.r)])
        elif isinstance(primitive, PolarShell):
            materials.append(
                _MaterialInterval(
                    float(primitive.r_in),
                    float(primitive.r_out),
                    float(primitive.rho),
                    str(primitive.material),
                    "PolarShell",
                )
            )
            primitive_interfaces.extend(
                [float(primitive.r_in), float(primitive.r_out)]
            )

    domain_start = (
        float(base.s_inner) if base.center_treatment == "annular" else 0.0
    )
    segment_start = domain_start
    if base.multiblock_r_match is not None:
        segment_start = float(base.multiblock_r_match)

    intervals = _compose_intervals(
        materials, segment_start, float(base.s_max), void_rho
    )
    mesh_kwargs: Dict = {
        "logical_mesh_2d": "spherical_polar_halfplane",
        "polar_center_treatment": str(base.center_treatment),
        "spherical_polar_s_max": float(base.s_max),
    }
    report: Dict = {
        "directions": {},
        "total_cells": None,
        "warnings": [],
        "interfaces_pinned": [],
    }
    direction_notes: List[str] = []
    smooth_records: List[_SmoothRegion] = []

    if smooth is not None:
        (
            nodes,
            counts,
            smooth_report,
            smooth_warnings,
            smooth_records,
        ) = _smooth_plan_intervals(intervals, void_rho, quality, smooth, 3)
        mesh_kwargs["explicit_nodes"] = nodes
        mesh_kwargs["nr"] = len(nodes) - 1
        report["smooth"] = smooth_report
        report["warnings"].extend(smooth_warnings)
    else:
        counts = _allocate_counts(
            intervals,
            int(quality.n_radial_hint),
            int(quality.default_zones_per_region),
            3,
        )
        use_segments = (
            base.center_treatment == "annular"
            or base.multiblock_r_match is not None
        )
        if use_segments:
            mesh_kwargs["grid_r"] = {
                "type": "graded",
                "segments": _segments(intervals, counts),
            }
            mesh_kwargs["nr"] = int(sum(counts))
        else:
            mesh_kwargs["auto_regions"] = _auto_regions(
                intervals, counts, void_rho
            )
            mesh_kwargs["auto_zone"] = {
                "mass_ratio_max": float(quality.mass_ratio_max)
            }

    if base.multiblock_r_match is not None:
        mesh_kwargs["topology_scheme"] = (
            "multiblock_cart_core_polar_shell"
        )
        direction_notes.append("multiblock_keys_required")
        dropped = [
            value
            for value in _dedupe(primitive_interfaces)
            if value > domain_start + _tolerance(domain_start)
            and value <= segment_start + _tolerance(segment_start)
        ]
        if dropped:
            report["warnings"].append(
                "interface below r_match absorbed by core/bridge"
            )

    _record_direction(
        report, "r", _boundaries(intervals), counts, direction_notes
    )
    for cone in primitives:
        if isinstance(cone, Cone) and cone.exterior_only:
            mesh_kwargs["cone_theta_wall"] = float(cone.theta_half)
            report["cone_exterior"] = {
                "theta_wall": float(cone.theta_half),
                "skipped_global_refine": True,
            }
    _plan_theta(primitives, base, mesh_kwargs, report)
    if base.center_mode == "graded_button":
        _apply_graded_button(
            base, mesh_kwargs, report, nodes, smooth_records[0]
        )
    # Planner zone counts express refinement ACROSS segments; within-segment
    # distribution stays uniform to avoid the 1D edge_ratio=0.1 default
    # producing razor cells at segment edges.
    if "grid_r" in mesh_kwargs:
        mesh_kwargs["grid_r"]["grading"] = {
            "edge_ratio": 0.9999999999999999
        }
    elif "grid_theta" in mesh_kwargs:
        mesh_kwargs["grid_theta"]["grading"] = {
            "edge_ratio": 0.9999999999999999
        }
    _finish_total_cells(report)
    return MeshPlan(mesh_kwargs=mesh_kwargs, report=report)


def _rect_materials(primitives: List, direction: str) -> List[_MaterialInterval]:
    result: List[_MaterialInterval] = []
    for primitive in primitives:
        if direction == "z" and isinstance(primitive, SlabLayer):
            result.append(
                _MaterialInterval(
                    float(primitive.z_from),
                    float(primitive.z_to),
                    float(primitive.rho),
                    str(primitive.material),
                    "SlabLayer",
                )
            )
        elif direction == "r" and isinstance(primitive, RectRadialLayer):
            result.append(
                _MaterialInterval(
                    float(primitive.r_from),
                    float(primitive.r_to),
                    float(primitive.rho),
                    str(primitive.material),
                    "RectRadialLayer",
                )
            )
    return result


def _plan_rect(
    primitives: List,
    base: RectBase,
    quality: Quality,
    smooth: Optional[SmoothZoning] = None,
) -> MeshPlan:
    if smooth is not None and any(
        isinstance(primitive, RectRadialLayer) for primitive in primitives
    ):
        raise ValueError("smooth zoning does not support RectRadialLayer yet")

    void_rho = _void_density(primitives)
    slab_materials = _rect_materials(primitives, "z")
    radial_materials = _rect_materials(primitives, "r")
    mesh_kwargs: Dict = {
        "r_min": float(base.r_min),
        "r_max": float(base.r_max),
        "z_min": float(base.z_min),
        "z_max": float(base.z_max),
    }
    report: Dict = {
        "directions": {},
        "total_cells": None,
        "warnings": [],
        "interfaces_pinned": [],
    }

    if slab_materials:
        z_intervals = _compose_intervals(
            slab_materials,
            float(base.z_min),
            float(base.z_max),
            void_rho,
        )
        if smooth is not None:
            (
                nodes,
                z_counts,
                smooth_report,
                smooth_warnings,
                _,
            ) = (
                _smooth_plan_intervals(
                    z_intervals, void_rho, quality, smooth, 1
                )
            )
            mesh_kwargs["explicit_nodes_z"] = nodes
            mesh_kwargs["nz"] = len(nodes) - 1
            report["smooth"] = smooth_report
            report["warnings"].extend(smooth_warnings)
        else:
            z_counts = _allocate_counts(
                z_intervals,
                int(quality.n_z_hint),
                int(quality.default_zones_per_region),
                1,
            )
            mesh_kwargs["auto_regions"] = _auto_regions(
                z_intervals, z_counts, void_rho
            )
            mesh_kwargs["auto_regions_axis"] = "z"
            mesh_kwargs["auto_zone"] = {
                "mass_ratio_max": float(quality.mass_ratio_max)
            }
        _record_direction(
            report, "z", _boundaries(z_intervals), z_counts
        )

    if radial_materials:
        r_intervals = _compose_intervals(
            radial_materials,
            float(base.r_min),
            float(base.r_max),
            void_rho,
        )
        r_counts = _allocate_counts(
            r_intervals,
            int(quality.n_radial_hint),
            int(quality.default_zones_per_region),
            2,
        )
        if slab_materials:
            mesh_kwargs["grid_r"] = {
                "type": "graded",
                "segments": _segments(r_intervals, r_counts),
            }
            mesh_kwargs["nr"] = int(sum(r_counts))
        else:
            mesh_kwargs["auto_regions"] = _auto_regions(
                r_intervals, r_counts, void_rho
            )
            mesh_kwargs["auto_regions_axis"] = "r"
            mesh_kwargs["auto_zone"] = {
                "mass_ratio_max": float(quality.mass_ratio_max)
            }
        _record_direction(
            report, "r", _boundaries(r_intervals), r_counts
        )

    _finish_total_cells(report)
    return MeshPlan(mesh_kwargs=mesh_kwargs, report=report)


def plan_mesh(
    primitives: List,
    base,
    quality: Quality = None,
    smooth: SmoothZoning = None,
) -> MeshPlan:
    """Compose primitives into namelist-representable TENRYU Mesh kwargs."""

    quality = quality or Quality()
    _validate_primitives(primitives, base)
    if isinstance(base, PolarBase):
        return _plan_polar(primitives, base, quality, smooth)
    if isinstance(base, RectBase):
        return _plan_rect(primitives, base, quality, smooth)
    raise ValueError(f"unsupported base {type(base).__name__}")
