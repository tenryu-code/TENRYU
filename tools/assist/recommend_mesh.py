"""Mesh block construction and optional solver validation for the recommender."""

from __future__ import annotations

import ast
import bisect
import heapq
import copy
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from tools.assist.mesh_recommendation import CampaignModel, INTERIOR, VOID_DENSITY

REPO_ROOT = Path(__file__).resolve().parents[2]


def layer_layout(conditions):
    position = conditions['r_min_cm']
    layout = []
    for layer in conditions['layers']:
        end = position + layer['thickness_cm']
        layout.append((position, end, layer['rho_gcc']))
        position = end
    return layout


def material_layout(conditions):
    """layer_layout without the void layers: the target the requirement depths measure."""
    return [span for span, layer in zip(layer_layout(conditions), conditions['layers']) if not layer.get('void')]


def column_at(layout, radius):
    return sum(max(0, min(radius, hi) - lo) * rho for lo, hi, rho in layout)


def measure_parameters(geometry):
    return {'planar': ('areal_mass', 1, 1.0),
            'cylindrical': ('cylindrical_line_mass', 2, 2 * math.pi),
            'spherical': ('spherical_cell_mass', 3, 4 * math.pi)}[geometry]


def cell_measure(geometry, rho, lo, hi):
    _, power, factor = measure_parameters(geometry)
    return rho * factor * (hi - lo) * sum(hi**(power - 1 - j) * lo**j for j in range(power)) / power


def surface_boundary(conditions):
    """Inner radius of the campaign's laser-side half of the target mass.

    Planar targets: half of all material layers; spherical and cylindrical targets: half
    of the outermost (shell) layer, as the campaign ladder refined it.
    """
    layout = material_layout(conditions)
    target = layout[-1:] if conditions['geometry'] != 'planar' else layout
    geometry = conditions['geometry']
    _, power, factor = measure_parameters(geometry)
    remaining = sum(cell_measure(geometry, rho, lo, hi) for lo, hi, rho in target) / 2
    for lo, hi, rho in target:
        mass = cell_measure(geometry, rho, lo, hi)
        if remaining <= mass:
            return (lo**power + power * remaining / (factor * rho))**(1 / power)
        remaining -= mass
    raise ValueError('cannot locate surface half')


# Hard caps carry this headroom below the ceiling they enforce.
HEADROOM = 0.95
# Spherical and cylindrical targets: an innermost layer lighter than this fraction of the
# densest target layer is gas fill, zoned by a cell count instead of a resolution cap.
FILL_DENSITY_FRACTION = 0.1
# Cells in an unbanded fill or void segment (the campaign value); also the solver's
# min_cells_per_segment for every segment.
SEGMENT_CELLS = 40
# Radius ratio of the payload sub-bands that express the local (rho * dr) rule in the
# spherical and cylindrical cell-measure: at most 1.2**2 = 1.44 conservative in area.
SUBBAND_RADIUS_RATIO = 1.2
# Adjacent-cell measure ratio of the emitted zoning.
RATIO = 1.3
# Solver band radii below this fraction of the target radius are the centre: the
# depth-to-radius inversion leaves a cube-root round-off residue there (about 5e-6 R0).
CENTRE_RADIUS_FRACTION = 1e-4
# Interval edges closer than this fraction of the target thickness merge (the scale at
# which mesh_regions snaps the ablation zone onto layer edges).
EDGE_MERGE_FRACTION = 1e-6


def reference_area(geometry, radius):
    _, power, factor = measure_parameters(geometry)
    return factor * radius**(power - 1)


def density_regions(conditions):
    """(r_end, rho) of the zoning density regions the recommendation emits: the layers,
    then the void padding to r_max."""
    regions = [(hi, rho) for lo, hi, rho in layer_layout(conditions)]
    if conditions['r_max_cm'] > regions[-1][0]:
        regions.append((conditions['r_max_cm'], VOID_DENSITY))
    return regions


def payload_core_radius(conditions, a, r_hi):
    """Largest radius c <= r_hi with c * max(rho on [r_min, c]) <= a / 2 (builder.cpp).

    Every cell inside the core meets the local rule rho * dr <= a without a bound.
    """
    rho_max, begin = 0.0, conditions['r_min_cm']
    for end, rho in density_regions(conditions):
        if not end > begin:
            continue
        rho_max = max(rho_max, rho)
        limit = .5 * a / rho_max if rho_max > 0 else math.inf
        if limit < min(end, r_hi):
            return max(begin, limit)
        if not end < r_hi:
            return r_hi
        begin = end
    limit = .5 * a / rho_max if rho_max > 0 else math.inf
    return min(r_hi, max(begin, limit))


def payload_pieces(conditions, lo, hi, a):
    """(r_lo, r_hi, conversion radius, factor) of a payload band, as builder.cpp splits it.

    A core [lo, c] bounded by area(c) * a / p (it admits the whole core as one cell and
    keeps rho * dr <= a for a cell reaching past c), then pieces of radius ratio <= 1.2
    bounded at their inner radius.
    """
    _, power, _ = measure_parameters(conditions['geometry'])
    pieces = []
    r = lo
    core = payload_core_radius(conditions, a, hi)
    if not r > 0 or r < core:
        if core > r:
            pieces.append((r, core, core, 1.0 / power))
        r = max(r, core)
    while r > 0 and r < hi:
        end = min(hi, r * SUBBAND_RADIUS_RATIO)
        pieces.append((r, end, r, 1.0))
        r = end
    return pieces


def target_depth(conditions, radius):
    """Requirement depth [g/cm^2]: target mass outside `radius` per reference area.

    The reference area is that of the outer target radius R0 (4 pi R0^2, 2 pi R0 or 1), as
    in mesh_requirement_cell_areal_mass; void layers carry no depth, as in the solver.
    """
    geometry = conditions['geometry']
    layout = material_layout(conditions)
    mass = sum(cell_measure(geometry, rho, max(lo, radius), hi)
               for lo, hi, rho in layout if hi > radius)
    return mass / reference_area(geometry, layout[-1][1])


def radius_at_depth(conditions, depth):
    """Inverse of target_depth on the piecewise-constant material layers."""
    geometry = conditions['geometry']
    _, power, factor = measure_parameters(geometry)
    layout = material_layout(conditions)
    remaining = depth * reference_area(geometry, layout[-1][1])
    for lo, hi, rho in reversed(layout):
        mass = cell_measure(geometry, rho, lo, hi)
        if remaining <= mass:
            inner = hi**power - power * remaining / (factor * rho)
            # Round-off in inner**(1/power) near the layer's inner edge (a cube root
            # turns 1e-21 into 1e-7) must not leave a sliver interval.
            if inner <= lo**power + 1e-9 * hi**power:
                return lo
            return inner**(1 / power)
        remaining -= mass
    return layout[0][0]


def ablation_zone_depth(conditions, recommendation, state=None):
    """Depth zoned at the surface ceiling.

    The predicted ablated depth; for learned recommendations (measured case or local
    trend) at least the campaign's laser-side half of the target, the region its ladder
    refined; never beyond the target. `state` carries depths enlarged by validation.
    """
    layout = material_layout(conditions)
    depth = recommendation['mu_abl_total_g_cm2']
    if recommendation['mode'] in ('measured_case', 'local_trend'):
        depth = max(depth, target_depth(conditions, surface_boundary(conditions)))
    if state:
        depth = max(depth, state.get('ablation_depth_g_cm2', 0.0))
    return min(depth, target_depth(conditions, layout[0][0]))


def mesh_regions(conditions, recommendation, state=None):
    """Radial regions: fill (unbanded gas), payload (local rule), ablation (surface cap).

    Void layers belong to none of them: the payload starts at the innermost material
    layer (after a fill), and a gas fill is a light innermost layer that is material.
    """
    geometry = conditions['geometry']
    layout = layer_layout(conditions)
    material = material_layout(conditions)
    r_min, outer = layout[0][0], layout[-1][1]
    depth = ablation_zone_depth(conditions, recommendation, state)
    r_ablation = radius_at_depth(conditions, depth)
    # A zone edge within a round-off sliver of a layer edge snaps onto it, so no
    # vanishing payload or fill interval (with a degenerate cap) is emitted.
    sliver = 1e-6 * (outer - r_min)
    for edge in [lo for lo, _, _ in layout] + [outer]:
        if abs(r_ablation - edge) <= sliver:
            r_ablation = edge
            depth = target_depth(conditions, edge)
            break
    fill = None
    if geometry != 'planar' and len(material) > 1 and not conditions['layers'][0].get('void'):
        lo, hi, rho = layout[0]
        if rho < FILL_DENSITY_FRACTION * max(rho for _, _, rho in material) and min(hi, r_ablation) > lo:
            fill = (lo, min(hi, r_ablation))
    payload_lo = fill[1] if fill is not None else material[0][0]
    payload = (payload_lo, r_ablation) if r_ablation > payload_lo else None
    return dict(depth=depth, r_ablation=r_ablation, r_min=r_min, outer=outer,
                fill=fill, payload=payload, ablation=(r_ablation, outer))


def interior_areal_mass(requirement):
    """Local rho * dr cap of unablated material: the campaign interior value, or the
    requirement's shock-separation ceiling when that is finer."""
    interior, basis = INTERIOR, 'C02-S1: 2e-5 g/cm2 changed campaign observables by <=1%'
    shocks = (requirement or {}).get('shocks') if isinstance(requirement, dict) else None
    if isinstance(shocks, dict) and shocks.get('applicable') is True:
        ceiling = shocks.get('ceiling_g_cm2')
        if isinstance(ceiling, (int, float)) and math.isfinite(ceiling) and 0 < ceiling < interior:
            interior, basis = ceiling, 'requirement shock-separation ceiling (finer than the campaign interior 2e-5 g/cm2)'
    return interior, basis


def own_band_pieces(conditions, recommendation, regions, interior):
    """Hard cell-measure caps (r_lo, r_hi, cap, kind) the recommender emits.

    Ablation zone: the ablation rule's reference areal mass, a constant cell-measure cap
    (reference area of R0). Payload: the local rule rho * dr <= interior on the pieces of
    payload_pieces (a core, then radius ratio <= 1.2).
    """
    geometry = conditions['geometry']
    area_ref = reference_area(geometry, regions['outer'])
    pieces = []
    if regions['payload'] is not None:
        lo, hi = regions['payload']
        if geometry == 'planar':
            pieces.append((lo, hi, HEADROOM * interior, 'payload'))
        else:
            for a, b, radius, factor in payload_pieces(conditions, lo, hi, interior):
                kind = 'payload_core' if factor < 1.0 else 'payload'
                pieces.append((a, b, HEADROOM * reference_area(geometry, radius) * interior * factor, kind))
    lo, hi = regions['ablation']
    pieces.append((lo, hi, HEADROOM * area_ref * recommendation['surface_areal_mass_g_cm2'], 'ablation'))
    return pieces


def solver_band_pieces(conditions, requirement):
    """Caps the solver's enforce path will inject, converted as builder.cpp does.

    Ablation-rule bands: the reference area of R0. Payload bands (local rho * dr rule) in
    the spherical and cylindrical measures: the pieces of payload_pieces.
    """
    if not isinstance(requirement, dict) or requirement.get('applicable') is not True:
        return []
    geometry = conditions['geometry']
    outer = layer_layout(conditions)[-1][1]
    reference = (requirement.get('inputs') or {}).get('R0_cm')
    if not (isinstance(reference, (int, float)) and math.isfinite(reference) and reference > 0):
        reference = outer
    pieces = []
    for band in requirement.get('bands_recommended') or []:
        values = [band.get(k) for k in ('areal_mass_max_g_cm2', 'r_lo_cm', 'r_hi_cm')]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
            continue
        ceiling, lo, hi = values
        if ceiling <= 0 or hi <= lo:
            continue
        kind = 'solver_' + str(band.get('kind'))
        if band.get('kind') != 'payload' or geometry == 'planar':
            pieces.append((lo, hi, reference_area(geometry, reference) * ceiling, kind))
            continue
        if lo <= CENTRE_RADIUS_FRACTION * reference:
            # The round-off residue of the solver's depth-to-radius inversion: the band
            # starts at the centre (its measure fraction there is zero either way).
            lo = conditions['r_min_cm']
        for a, b, radius, factor in payload_pieces(conditions, lo, hi, ceiling):
            pieces.append((a, b, reference_area(geometry, radius) * ceiling * factor, kind))
    return [piece for piece in pieces if piece[2] > 0 and math.isfinite(piece[2])]


# The solver's zoning quadrature (src/core/zoning_intent.cpp): positions closer than
# 1e-12 of the domain length coincide, and each panel between profile knots is refined
# from 8 to 16384 bins until its measure and monitor totals change by <= 1e-12 and every
# bin's Simpson-trapezoid difference is <= 1e-10 of the panel total.
POSITION_TOLERANCE_SCALE = 1e-12
INTEGRAL_TOLERANCE = 1e-12
BIN_TOLERANCE = 1e-10
MAXIMUM_BINS = 16384


def profile_weight_function(profile):
    """Log-linear profile weight with constant extrapolation, as the solver evaluates it."""
    radii = [p['r'] for p in profile]
    logs = [math.log(p['w']) for p in profile]
    def weight(r):
        if not radii:
            return 1.0
        if r <= radii[0]:
            return math.exp(logs[0])
        if r >= radii[-1]:
            return math.exp(logs[-1])
        j = bisect.bisect_right(radii, r)
        fraction = (r - radii[j - 1]) / (radii[j] - radii[j - 1])
        return math.exp(logs[j - 1] + fraction * (logs[j] - logs[j - 1]))
    return weight


def _quadrature_panel(geometry, rho, a, b, bins, weight):
    _, power, factor = measure_parameters(geometry)
    h = (b - a) / bins
    coordinates = [a + k * h for k in range(bins + 1)]
    coordinates[0], coordinates[-1] = a, b
    def sample(r):
        measure = factor * rho * r**(power - 1)
        return measure, measure / weight(r)
    values = [sample(math.nextafter(a, b) if k == 0 else math.nextafter(b, a) if k == bins else coordinates[k])
              for k in range(bins + 1)]
    mids = [sample((coordinates[k] + coordinates[k + 1]) / 2.0) for k in range(bins)]
    tables = []
    for j in (0, 1):
        total, difference = 0.0, 0.0
        for k in range(bins):
            step = coordinates[k + 1] - coordinates[k]
            simpson = step / 6.0 * (values[k][j] + 4.0 * mids[k][j] + values[k + 1][j])
            total += simpson
            difference = max(difference, abs(simpson - step / 2.0 * (values[k][j] + values[k + 1][j])))
        tables.append((total, difference))
    return tables


def monitor_integral(geometry, rho, lo, hi, profile, position_tolerance):
    """Segment monitor total by the solver's panelwise Simpson quadrature."""
    edges = sorted([lo, hi] + [p['r'] for p in profile if lo < p['r'] < hi])
    panels = [edges[0]]
    for edge in edges[1:]:
        if edge - panels[-1] > position_tolerance:
            panels.append(edge)
    panels[-1] = hi
    weight = profile_weight_function(profile)
    total = 0.0
    for a, b in zip(panels, panels[1:]):
        previous = _quadrature_panel(geometry, rho, a, b, 4, weight)
        bins = 8
        while True:
            current = _quadrature_panel(geometry, rho, a, b, bins, weight)
            if all(abs(cur[0] - old[0]) <= INTEGRAL_TOLERANCE * max(abs(cur[0]), 1e-300) and
                   cur[1] <= BIN_TOLERANCE * max(cur[0], 1e-300) for cur, old in zip(current, previous)):
                break
            if bins >= MAXIMUM_BINS:
                raise ValueError('mesh budget monitor quadrature did not converge on panel [{0!r}, {1!r}]'.format(a, b))
            previous, bins = current, bins * 2
        total += current[1][0]
    return total


def allocate_segments(n_cells, monitors, minimum=40):
    """Solver largest-deficit rule; ties prefer the lower segment index."""
    total = sum(monitors)
    ideal = [n_cells * m / total for m in monitors]
    counts = [minimum] * len(monitors)
    if n_cells < sum(counts):
        raise ValueError('cell budget is below segment minima')
    heap = [(counts[i] - ideal[i], i) for i in range(len(counts))]
    heapq.heapify(heap)
    for _ in range(n_cells - sum(counts)):
        _, i = heapq.heappop(heap)
        counts[i] += 1
        heapq.heappush(heap, (counts[i] - ideal[i], i))
    return counts, ideal


def build_mesh(payload, n_cells=None, dr_min=None, empirical_supported=True, margin=1.10, requirement=None):
    c, rec = payload['conditions'], payload['recommendation']
    state = payload.get('mesh_state') or {}
    if requirement is None:
        requirement = payload.get('solver_requirement')
    geometry = c['geometry']
    measure, power, factor = measure_parameters(geometry)
    layout = layer_layout(c)
    regions = mesh_regions(c, rec, state)
    outer = regions['outer']
    interior, interior_basis = interior_areal_mass(requirement)
    own = own_band_pieces(c, rec, regions, interior)
    solver = solver_band_pieces(c, requirement)
    segments = list(layout)
    voids = [bool(layer.get('void')) for layer in c['layers']]
    density = [dict(r_end=hi, rho=rho) for lo, hi, rho in layout]
    if c['r_max_cm'] > outer:
        density.append(dict(r_end=c['r_max_cm'], rho=VOID_DENSITY))
        segments.append((outer, c['r_max_cm'], VOID_DENSITY))
        voids.append(True)
    def cumulative(r):
        return sum(cell_measure(geometry, rho, lo, min(hi, r)) for lo, hi, rho in segments if lo < r)
    total = cumulative(c['r_max_cm'])
    position_tolerance = POSITION_TOLERANCE_SCALE * (c['r_max_cm'] - c['r_min_cm'])
    # Intervals between every layer, region and band edge; each has one preferred measure.
    # A solver band edge and a region edge computed independently can coincide up to
    # round-off; edges within EDGE_MERGE_FRACTION of the target thickness merge (layer and
    # domain edges win), so no sliver interval puts profile knots within the solver's
    # position tolerance. Hard caps keep their exact radii in the bands below.
    merge = EDGE_MERGE_FRACTION * (outer - c['r_min_cm'])
    edges = sorted(set([seg[0] for seg in segments] + [c['r_max_cm']]))
    for r in sorted(set([p[k] for p in own + solver for k in (0, 1)] +
                        [r for r in (regions['r_ablation'],) + (regions['fill'] or ()) if r is not None])):
        if c['r_min_cm'] <= r <= c['r_max_cm'] and all(abs(r - e) > merge for e in edges):
            edges.append(r)
    edges.sort()
    fill_mass = (cumulative(regions['fill'][1]) - cumulative(regions['fill'][0])) if regions['fill'] else 0.0
    intervals = []
    for a, b in zip(edges, edges[1:]):
        if b <= a:
            continue
        segment = next(i for i, (lo, hi, rho) in enumerate(segments) if lo <= a and b <= hi * (1 + 1e-15) + 1e-300)
        rho = segments[segment][2]
        mass = cell_measure(geometry, rho, a, b)
        caps = [cap for lo, hi, cap, kind in own + solver if lo < b and hi > a]
        if voids[segment]:
            # The padding beyond the target and void layers inside it: the per-segment
            # minimum of cells, placed by the constant or neighbouring profile weight.
            lo, hi, _ = segments[segment]
            kind, weight, count = 'void', cell_measure(geometry, rho, lo, hi) / SEGMENT_CELLS, 0
        elif caps:
            kind, weight = 'capped', min(caps)
            count = math.ceil(mass / weight)
        else:
            kind, weight, count = 'fill', max(fill_mass, mass) / SEGMENT_CELLS, 0
        intervals.append(dict(r_lo_cm=a, r_hi_cm=b, segment=segment, rho_gcc=rho,
                              measure=mass, preferred_measure=weight, cap=weight, kind=kind, cells=count))
    # Per-segment cell requirement: capped counts, ratio-limited transitions between
    # neighbouring intervals of one segment, and the solver's per-segment minimum.
    requirements = []
    for index in range(len(segments)):
        own_intervals = [item for item in intervals if item['segment'] == index]
        cells = sum(item['cells'] for item in own_intervals)
        transitions = 0
        for left, right in zip(own_intervals, own_intervals[1:]):
            ratio = max(left['cap'], right['cap']) / min(left['cap'], right['cap'])
            if ratio > RATIO:
                transitions += math.ceil(math.log(ratio) / math.log(RATIO))
        requirements.append((cells, transitions, max(SEGMENT_CELLS, cells + transitions + 2)))
    # Void intervals (the padding and void layers) stay out of the profile: the profile ends
    # at the target's outer radius, and the constant or neighbouring weight gives a void a
    # negligible monitor, so it keeps the per-segment minimum (the campaign's padding), and
    # no ramp toward a void's tiny measure enters the target.
    target_intervals = [item for item in intervals if item['kind'] != 'void']
    profile = []
    def point(r, w):
        if profile and r <= profile[-1]['r']:
            profile[-1]['w'] = w
        else:
            profile.append(dict(r=r, w=w))
    def build_profile():
        profile.clear()
        for i, item in enumerate(target_intervals):
            if i and item['preferred_measure'] != target_intervals[i - 1]['preferred_measure']:
                left = profile[-1]['r']
                offset = max(max(abs(item['r_lo_cm']), item['r_lo_cm'] - left) * 1e-6, 1e3 * position_tolerance)
                point(item['r_lo_cm'] - min((item['r_lo_cm'] - left) * .5, offset),
                      target_intervals[i - 1]['preferred_measure'])
            point(item['r_lo_cm'], item['preferred_measure'])
        point(target_intervals[-1]['r_hi_cm'], target_intervals[-1]['preferred_measure'])
        return [monitor_integral(geometry, rho, lo, hi, profile, position_tolerance) for lo, hi, rho in segments]
    monitors = build_profile()
    # The solver allocates cells by monitor share: segments at the per-segment minimum get
    # it regardless of their share, and the cells they take above their share come out of
    # the other segments in equal absolute amounts (largest deficit), at most
    # SEGMENT_CELLS per minimum segment in total. A segment whose monitor falls below its
    # requirement plus that amount (hard caps far above its cells' measures, as over light
    # material) would force the total up in proportion; its preferred measures scale down
    # uniformly instead, which leaves the cell placement within the segment unchanged.
    scales = [1.0] * len(segments)
    allowance = SEGMENT_CELLS * sum(1 for _, _, required in requirements if required <= SEGMENT_CELLS)
    for index, (cells, transitions, required) in enumerate(requirements):
        target = required + allowance
        if required > SEGMENT_CELLS and segments[index][0] < outer and monitors[index] < target:
            scales[index] = monitors[index] / target
    if any(scale < 1.0 for scale in scales):
        for item in intervals:
            item['preferred_measure'] *= scales[item['segment']]
        monitors = build_profile()
    bands = [dict(measure_frac_begin=cumulative(lo) / total, measure_frac_end=cumulative(hi) / total,
                  cell_measure_max=cap) for lo, hi, cap, kind in own]
    audits = []
    for index, (lo, hi, rho) in enumerate(segments):
        cells, transitions, required = requirements[index]
        audits.append(dict(r_lo_cm=lo, r_hi_cm=hi, measure_total=cell_measure(geometry, rho, lo, hi),
                           monitor_total=monitors[index], capped_cells=cells, transition_cells=transitions,
                           required_cells=required, profile_scale=scales[index]))
    base = sum(a['required_cells'] for a in audits)
    count = max(math.ceil(margin * base), n_cells or 0)
    for _ in range(200):
        counts, ideal = allocate_segments(count, monitors, SEGMENT_CELLS)
        deficits = [a['required_cells'] - n for a, n in zip(audits, counts)]
        if max(deficits) <= 0:
            break
        count += max(1, max(math.ceil(d * sum(monitors) / m) for d, m in zip(deficits, monitors) if d > 0))
    else:
        raise ValueError('could not satisfy segment allocation budget')
    width = math.inf
    for (lo, hi, rho), n, audit, share in zip(segments, counts, audits, ideal):
        audit.update(allocated_cells=n, ideal_share=share)
        width = min(width, audit['measure_total'] / n / (rho * reference_area(geometry, hi)))
    for item in intervals:
        if item['kind'] == 'capped':
            width = min(width, item['cap'] / (item['rho_gcc'] * reference_area(geometry, item['r_hi_cm'])))
    floor = .05 * width
    if dr_min is not None:
        floor = min(floor, dr_min)
    rr = dict(apply='enforce')
    if empirical_supported and rec['empirical'] is not None:
        rr['empirical'] = rec['empirical']
    pins = [dict(r=hi, ratio_jump_allowed=True) for lo, hi, rho in layout if hi < c['r_max_cm']]
    mesh = dict(r_min=c['r_min_cm'], r_max=c['r_max_cm'], geometry_1d=geometry,
                zoning_intent=dict(n_cells=count, measure=measure, density_regions=density,
                    pins=pins, profile=profile, bands=bands, dr_min=floor,
                    preferred_ratio=RATIO, ratio_hard_max=RATIO, min_cells_per_segment=SEGMENT_CELLS),
                resolution_requirement=rr)
    return mesh, dict(estimated_n_cells=count, n_cells=count, dr_min_cm=floor, margin=margin,
                      base_required_cells=base, segments=audits,
                      band_regions=[dict(r_lo_cm=lo, r_hi_cm=hi, cell_measure_max=cap, kind=kind) for lo, hi, cap, kind in own],
                      solver_band_regions=[dict(r_lo_cm=lo, r_hi_cm=hi, cell_measure_max=cap, kind=kind) for lo, hi, cap, kind in solver],
                      ablation_r_lo_cm=regions['r_ablation'], ablation_depth_g_cm2=regions['depth'],
                      campaign_r_lo_cm=surface_boundary(c), fill_region_cm=regions['fill'],
                      payload_region_cm=regions['payload'], mu_abl_total_g_cm2=rec['mu_abl_total_g_cm2'],
                      interior_areal_mass_g_cm2=interior, interior_basis=interior_basis,
                      estimate_basis='Per-interval counts at the hard caps (ablation zone: reference-area ceiling; payload: local rho*dr rule on a core and radius-ratio-1.2 pieces; solver-injected caps folded in when a probe preview exists), ratio-limited transitions, 40 cells per unbanded fill/void segment; a segment whose requirement exceeds its monitor has its preferred measures scaled down to match; margin times the sum, increased until the largest-deficit allocation meets every segment')


def mesh_block(mesh):
    # JSON formatting is deterministic; convert only JSON booleans/null outside strings.
    def literal(value):
        if isinstance(value, dict):
            return '{' + ', '.join(repr(k) + ': ' + literal(v) for k, v in value.items()) + '}'
        if isinstance(value, list):
            return '[' + ', '.join(literal(v) for v in value) + ']'
        return repr(value)
    lines = ['Mesh(']
    for key, value in mesh.items():
        if isinstance(value, dict):
            lines.append('    ' + key + '=dict(')
            lines.extend('        ' + k + '=' + literal(v) + ',' for k, v in value.items())
            lines.append('    ),')
        else:
            lines.append('    ' + key + '=' + literal(value) + ',')
    lines.append(')')
    return '\n'.join(lines) + '\n'


def replace_mesh(deck, block):
    tree = ast.parse(deck)
    calls = [node for node in tree.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
             and isinstance(node.value.func, ast.Name) and node.value.func.id == 'Mesh']
    if len(calls) != 1:
        raise ValueError('deck must contain exactly one top-level Mesh(...) call')
    node = calls[0]
    replaced = {'nr', 'r_min', 'r_max', 'geometry_1d', 'grid', 'grading',
                'auto_regions', 'auto_zone', 'zoning_intent', 'resolution_requirement',
                'explicit_nodes'}
    preserved = []
    for keyword in node.value.keywords:
        if keyword.arg is None:
            raise ValueError('cannot replace a Mesh call with **kwargs; expand its mesh keys first')
        if keyword.arg not in replaced:
            preserved.append('    ' + keyword.arg + '=' + ast.get_source_segment(deck, keyword.value) + ',\n')
    if preserved:
        end = block.rfind(')')
        block = block[:end] + ''.join(preserved) + block[end:]
    lines = deck.splitlines(keepends=True)
    return ''.join(lines[:node.lineno - 1]) + block + ''.join(lines[node.end_lineno:])


def literal_conditions(deck):
    """Offline extraction is deliberately limited; never execute the deck to guess."""
    for node in ast.parse(deck).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'MESH_EXPERIMENTAL_CONDITIONS' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('without a capable binary, --deck needs a literal MESH_EXPERIMENTAL_CONDITIONS dictionary; otherwise use --conditions JSON')


def conditions_from_preview(preview):
    requirement = preview.get('mesh_requirement', {})
    source = requirement.get('experimental_conditions')
    if not isinstance(source, dict):
        raise ValueError('binary did not export experimental_conditions; use --conditions JSON or a literal MESH_EXPERIMENTAL_CONDITIONS deck declaration')
    source = copy.deepcopy(source)
    # The solver already exports A/Z in inputs.materials, independently of the
    # compact layer descriptors. Use that authoritative data without a C++ change.
    materials = {m['name']: m for m in requirement.get('inputs', {}).get('materials', [])}
    for layer in source.get('layers', []):
        material = materials.get(layer['material'], {})
        if material.get('is_void') is True:
            # Vacuum inside the target; the solver exports void materials with Z = 0.
            layer['void'] = True
            continue
        for key in ('A', 'Z'):
            if key in material:
                layer[key] = material[key]
            elif key not in layer:
                raise ValueError('binary did not export material A and Z for ' + layer['material'] + '; use --conditions JSON')
    return source


def validate(deck_path, tenryu):
    from tools.assist.deck_lint import parse_mesh_preview
    run = subprocess.run([tenryu, 'validate', str(deck_path), '--mesh-preview'], cwd=str(REPO_ROOT),
                         capture_output=True, text=True, timeout=600)
    combined = run.stdout + run.stderr
    preview = parse_mesh_preview(combined)
    return run.returncode, preview, combined


def achieved_surface(preview, conditions, recommendation=None, state=None):
    """Largest reference areal mass [g/cm^2] of a preview cell in the zone that must meet
    the surface ceiling: the ablation zone of `recommendation` (the campaign's laser-side
    half without one). Cells straddling the zone edge count with their whole measure."""
    nodes = preview.get('r_nodes')
    if not isinstance(nodes, list) or len(nodes) < 2:
        return None
    if any(not isinstance(r, (int, float)) or not math.isfinite(r) for r in nodes) or any(b <= a for a, b in zip(nodes, nodes[1:])):
        return None
    layout = layer_layout(conditions)
    outer = layout[-1][1]
    if recommendation is None:
        boundary = surface_boundary(conditions)
    else:
        boundary = mesh_regions(conditions, recommendation, state)['r_ablation']
    area_ref = reference_area(conditions['geometry'], outer)
    maxima = []
    for lo, hi in zip(nodes, nodes[1:]):
        if hi > boundary and lo < outer:
            mass = sum(cell_measure(conditions['geometry'], rho, max(lo, a), min(hi, b))
                       for a, b, rho in layout if a < hi and b > lo)
            maxima.append(mass / area_ref)
    return max(maxima) if maxima else None


def compact_requirement(requirement):
    """The parts of a solver requirement preview that the mesh construction uses."""
    if not isinstance(requirement, dict) or requirement.get('applicable') is not True:
        return None
    shocks = requirement.get('shocks') if isinstance(requirement.get('shocks'), dict) else {}
    return dict(applicable=True, inputs=dict(R0_cm=(requirement.get('inputs') or {}).get('R0_cm')),
                shocks=dict(applicable=shocks.get('applicable'), ceiling_g_cm2=shocks.get('ceiling_g_cm2')),
                bands_recommended=[{k: band.get(k) for k in ('kind', 'r_lo_cm', 'r_hi_cm', 'areal_mass_max_g_cm2', 'width_max_cm')}
                                   for band in requirement.get('bands_recommended') or [] if isinstance(band, dict)])


def same_requirement(a, b, tolerance=1e-9):
    """Compact solver requirements equal up to round-off.

    The solver locates its band radii by summing the current mesh's cells, so one
    requirement probed on two meshes differs in the last digits of every radius, and a
    radius at the centre is the cube-root residue of that round-off (about 5e-6 R0):
    radii below CENTRE_RADIUS_FRACTION of the radius scale compare equal.
    """
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return a == b
    def number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    def close(x, y, scale):
        if number(x) and number(y):
            return (abs(x - y) <= tolerance * scale or
                    max(abs(x), abs(y)) <= CENTRE_RADIUS_FRACTION * scale)
        return x == y
    def relative(x, y):
        if number(x) and number(y):
            return abs(x - y) <= tolerance * max(abs(x), abs(y))
        return x == y
    bands_a, bands_b = a.get('bands_recommended') or [], b.get('bands_recommended') or []
    radii = [abs(v) for band in bands_a + bands_b for v in (band.get('r_lo_cm'), band.get('r_hi_cm')) if number(v)]
    radius_scale = max(radii + [abs(v) for v in ((a.get('inputs') or {}).get('R0_cm'), (b.get('inputs') or {}).get('R0_cm')) if number(v)] or [1.0])
    shocks_a, shocks_b = a.get('shocks') or {}, b.get('shocks') or {}
    return (a.get('applicable') == b.get('applicable') and
            close((a.get('inputs') or {}).get('R0_cm'), (b.get('inputs') or {}).get('R0_cm'), radius_scale) and
            shocks_a.get('applicable') == shocks_b.get('applicable') and
            relative(shocks_a.get('ceiling_g_cm2'), shocks_b.get('ceiling_g_cm2')) and
            len(bands_a) == len(bands_b) and
            all(x.get('kind') == y.get('kind') and
                close(x.get('r_lo_cm'), y.get('r_lo_cm'), radius_scale) and
                close(x.get('r_hi_cm'), y.get('r_hi_cm'), radius_scale) and
                relative(x.get('areal_mass_max_g_cm2'), y.get('areal_mass_max_g_cm2')) and
                relative(x.get('width_max_cm'), y.get('width_max_cm'))
                for x, y in zip(bands_a, bands_b)))


BUDGET_ERRORS = ('MESH_BAND_BOX_INFEASIBLE', 'MESH_SEGMENT_MIN_COUNT_INFEASIBLE',
                 'MESH_SEGMENT_BUDGET_CONFLICT', 'MESH_CHAIN_SUM_INFEASIBLE',
                 'MESH_CELL_MEASURE_BOX_INFEASIBLE', 'MESH_DR_MIN_COUNT_INFEASIBLE')


def parse_certificate(log):
    line = next((line for line in log.splitlines() if any(code in line for code in BUDGET_ERRORS + ('MESH_RESOLUTION_REQUIREMENT_VIOLATED',))), None)
    if line is None:
        return None
    result = dict(message=line)
    segment = re.search(r'\bsegment (\d+)', line)
    if segment:
        result['segment'] = int(segment.group(1))
    number = r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
    violation = re.search(r'(ablation|shock) worst cell (\d+) \[\s*' + number + r'\s*,\s*' + number +
                          r'\s*\] areal mass ' + number + r' > ceiling ' + number, line)
    if violation:
        result.update(rule=violation.group(1), cell=int(violation.group(2)),
                      r_lo_cm=float(violation.group(3)), r_hi_cm=float(violation.group(4)),
                      areal_mass_g_cm2=float(violation.group(5)), ceiling_g_cm2=float(violation.group(6)))
    for name, pattern in (
        ('sum_enabled_hi', r'sum\(enabled hi_i\)='), ('sum_enabled_lo', r'sum\(enabled lo_i\)='),
        ('measure_total', r'(?:measure_total=|total measure )'), ('Q_s', r'Q_s='), ('L_s', r'L_s='),
        ('N_min_s', r'N_min_s\)?='), ('N_max_s', r'N_max_s\)?='),
        ('n_cells', r'n_cells='), ('minimum_n_cells', r'(?:n_cells|N|count)\s*>=\s*'),
        ('capacity', r'capacity='), ('dr_min', r'dr_min='),
        ('lo_measure', r'lo_measure='), ('hi_measure', r'hi_measure='),
        ('lower_sum', r'lower_sum='), ('upper_sum', r'upper_sum=')):
        match = re.search(pattern + number, line)
        if match:
            result[name] = float(match.group(1))
    window = re.search(r'feasible window \[\s*' + number + r'\s*,\s*' + number + r'\s*\]', line)
    if window:
        result['feasible_min'] = float(window.group(1))
        result['feasible_max'] = float(window.group(2))
    return result


def validate_recommendation(payload, tenryu, deck_text, max_attempts=8):
    state = payload.setdefault('mesh_state', {})
    mesh, budget = build_mesh(payload)
    attempts = []
    supported = True
    def rebuild(**changes):
        current = mesh['zoning_intent']
        options = dict(n_cells=current['n_cells'], dr_min=current['dr_min'],
                       empirical_supported=supported, margin=budget['margin'])
        options.update(changes)
        return build_mesh(payload, **options)
    with tempfile.TemporaryDirectory(prefix='mesh_recommendation_') as directory:
        path = Path(directory) / 'candidate.py'
        for index in range(max_attempts):
            path.write_text(replace_mesh(deck_text, mesh_block(mesh)), encoding='utf-8')
            code, preview, log = validate(path, tenryu)
            attempt = dict(attempt=index + 1, n_cells=mesh['zoning_intent']['n_cells'],
                           dr_min_cm=mesh['zoning_intent']['dr_min'], exit_code=code,
                           margin=budget['margin'],
                           diagnostic_codes=sorted(set(re.findall(r'\bMESH_[A-Z0-9_]+', log))))
            error = next((line for line in log.splitlines() if re.search(
                r'MESH_EMPIRICAL_(?:INVALID|APRIORI_MISMATCH)', line)), None)
            if error is not None:
                attempt['error'] = error
            certificate = parse_certificate(log)
            if certificate is not None:
                attempt['certificate'] = certificate
            attempts.append(attempt)
            requirement = (preview or {}).get('mesh_requirement', {})
            if not isinstance(requirement, dict):
                requirement = {}
            compact = compact_requirement(requirement)
            if compact is not None and not same_requirement(payload.get('solver_requirement'), compact):
                # The solver's own injected caps, shock ceiling and reference radius now
                # enter the construction (a synthetic probe may differ from the deck).
                payload['solver_requirement'] = compact
                attempt['retry'] = dict(policy='fold_in_solver_requirement')
                mesh, budget = rebuild()
                continue
            preview_mu = (requirement.get('ablation') or {}).get('mu_abl_total_g_cm2')
            if isinstance(preview_mu, (int, float)) and math.isfinite(preview_mu) and preview_mu > payload['recommendation']['mu_abl_total_g_cm2'] * (1 + 1e-12):
                payload['recommendation'].update(mu_abl_total_g_cm2=preview_mu,
                                                mu_abl_source='solver mesh_requirement (conservative maximum)')
                attempt['retry'] = dict(policy='extend_to_solver_ablated_depth', mu_abl_total_g_cm2=preview_mu)
                mesh, budget = rebuild()
                continue
            check = requirement.get('requirement_check', {}) if isinstance(requirement.get('requirement_check'), dict) else {}
            ablation_violations = (check.get('ablation') or {}).get('n_violations', 0)
            shock_violations = (check.get('shock') or {}).get('n_violations', 0)
            surface = achieved_surface(preview or {}, payload['conditions'], payload['recommendation'], state)
            # Both rules must hold: the solver reports shock-rule violations of zoning_intent
            # decks without refusing them, and the margin ladder below refines such a mesh.
            if (code == 0 and ablation_violations == 0 and shock_violations == 0 and surface is not None and
                    surface <= payload['recommendation']['surface_areal_mass_g_cm2'] * (1 + 1e-6)):
                payload['validation'] = dict(status='validated', attempts=attempts, requirement_check=check,
                                             achieved_surface_areal_mass_g_cm2=surface)
                break
            if supported and 'MESH_EMPIRICAL_INVALID' in log and 'reference_sha256' in log:
                supported = False
                mesh['resolution_requirement'].pop('empirical', None)
                payload['flags'].append('reference_table_mismatch')
                payload['warnings'].append(
                    'Tool table SHA-256 ' + payload['reference_sha256'] +
                    ' differs from the table compiled into the binary. Calibrated enforce remains active '
                    'with explicit finer bands; the learned efficiency benefit is lost until the table '
                    'and binary agree. Use the same checkout state for tool, table and binary.')
                continue
            if supported and 'empirical' in log and re.search(r'unknown|unexpected|unrecognized', log, re.I):
                supported = False
                mesh['resolution_requirement'].pop('empirical', None)
                payload['flags'].append('legacy_binary')
                payload['warnings'].append('Binary rejects empirical keys; calibrated enforce remains active with explicit finer bands. The learned efficiency benefit may be lost.')
                continue
            if 'MESH_RESOLUTION_REQUIREMENT_DR_MIN_CONFLICT' in log:
                match = re.search(r'admissible dr_min <=\s*([0-9.eE+-]+)', log)
                if match:
                    mesh['zoning_intent']['dr_min'] = min(mesh['zoning_intent']['dr_min'] * .5, float(match.group(1)) * .5)
                    continue
            if 'MESH_EMPIRICAL_APRIORI_MISMATCH' in log:
                match = re.search(r'apriori_g_cm2=([0-9.eE+-]+)', log)
                if match:
                    solver_requirement = {'applicable': True, 'ablation': {'ceiling_formation_g_cm2': float(match.group(1))}}
                    revised = CampaignModel().recommend(payload['conditions'], solver_requirement)
                    revised['recommendation']['mu_abl_total_g_cm2'] = max(revised['recommendation']['mu_abl_total_g_cm2'], payload['recommendation']['mu_abl_total_g_cm2'])
                    payload['recommendation'] = revised['recommendation']
                    attempt['retry'] = dict(policy='recompute_with_solver_apriori')
                    mesh, budget = rebuild()
                    continue
            if certificate and certificate.get('rule') == 'ablation' and 'r_lo_cm' in certificate:
                depth = target_depth(payload['conditions'], certificate['r_lo_cm'])
                if depth > state.get('ablation_depth_g_cm2', 0.0) * (1 + 1e-12):
                    state['ablation_depth_g_cm2'] = depth
                    attempt['retry'] = dict(policy='extend_ablation_zone', ablation_depth_g_cm2=depth)
                    mesh, budget = rebuild()
                    continue
            if code == 0 or certificate is not None:
                current = mesh['zoning_intent']
                next_margin = next((m for m in (1.25, 1.45, 1.70) if m > budget['margin']), None)
                count = current['n_cells'] if next_margin is not None else math.ceil(current['n_cells'] * 1.5)
                parsed = certificate or {}
                count = max(count, math.ceil(parsed.get('minimum_n_cells', 0)),
                            math.ceil(parsed.get('N_min_s', 0)))
                floor = current['dr_min']
                if 'MESH_DR_MIN_COUNT_INFEASIBLE' in log or 'MESH_SEGMENT_BUDGET_CONFLICT' in log or 'N_max_s' in parsed:
                    floor *= .5
                attempt['retry'] = dict(margin=next_margin or budget['margin'],
                                        policy='campaign_margin' if next_margin is not None else 'count_times_1.5')
                mesh, budget = rebuild(n_cells=count, dr_min=floor, margin=next_margin or budget['margin'])
                continue
            attempt['reason'] = 'validation failed; inspect the deck with tenryu validate'
            break
        else:
            payload['warnings'].append('Validation retry limit reached.')
    if payload.get('validation', {}).get('status') != 'validated':
        payload['validation'] = dict(status='failed', attempts=attempts)
    budget.update(n_cells=mesh['zoning_intent']['n_cells'], dr_min_cm=mesh['zoning_intent']['dr_min'])
    payload.update(mesh=mesh, mesh_block=mesh_block(mesh), budget=budget)
    return payload


def recommendation_payload(source, requirement=None, model=None):
    model = CampaignModel() if model is None else model
    payload = model.recommend(source, requirement)
    compact = compact_requirement(requirement)
    if compact is not None:
        payload['solver_requirement'] = compact
    mesh, budget = build_mesh(payload)
    payload.update(mesh=mesh, mesh_block=mesh_block(mesh), budget=budget,
                   validation=dict(status='unvalidated'),
                   numerics_note='Merge Numerics(hydro=dict(driver_full_step_retry_enabled=True)) into the existing Numerics block; fine-mesh runtime companion.')
    return payload


def main_recommend_mesh(args):
    from tools.assist.deck_lint import _resolve_tenryu, _write_output
    try:
        requested = args.tenryu or os.environ.get('TENRYU_BIN')
        tenryu = _resolve_tenryu(requested)
        if requested and tenryu is None:
            raise ValueError('requested tenryu binary not found: ' + requested)
        deck = Path(args.deck).read_text(encoding='utf-8') if args.deck else None
        if args.deck_out and deck is None and tenryu is None:
            raise ValueError('--deck-out without --deck requires tenryu validation of a synthetic mesh-check deck')
        requirement = None
        if args.conditions:
            source = json.loads(Path(args.conditions).read_text(encoding='utf-8'))
        elif tenryu:
            code, preview, _ = validate(Path(args.deck).resolve(), tenryu)
            if preview:
                try:
                    source = conditions_from_preview(preview)
                except ValueError:
                    source = literal_conditions(deck)
                requirement = preview.get('mesh_requirement')
            else:
                source = literal_conditions(deck)
        else:
            source = literal_conditions(deck)
        payload = recommendation_payload(source, requirement)
        if tenryu:
            if deck is None:
                deck = conditions_deck(payload['conditions'])
            if requirement is None:
                # Obtain the real calibrated ceiling and depth without an empirical override.
                base_mesh = copy.deepcopy(payload['mesh'])
                base_mesh['resolution_requirement'] = dict(apply='report')
                with tempfile.TemporaryDirectory(prefix='mesh_requirement_probe_') as directory:
                    path = Path(directory) / 'probe.py'
                    path.write_text(replace_mesh(deck, mesh_block(base_mesh)), encoding='utf-8')
                    _, preview, _ = validate(path, tenryu)
                if preview and preview.get('mesh_requirement', {}).get('applicable'):
                    payload = recommendation_payload(source, preview['mesh_requirement'])
            payload = validate_recommendation(payload, tenryu, deck)
            if not args.deck:
                payload['validation']['deck_kind'] = 'synthetic_mesh_check'
                payload['warnings'].append('Conditions-only validation uses an ideal-gas material stub to check mesh geometry and requirement constraints; validate the production EOS deck separately.')
        if deck is not None:
            assembled = replace_mesh(deck, payload['mesh_block'])
        if args.deck:
            node = next(node for node in ast.parse(assembled).body
                        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Name) and node.value.func.id == 'Mesh')
            payload['mesh_block'] = ast.get_source_segment(assembled, node.value) + '\n'
        if args.deck_out:
            Path(args.deck_out).write_text(assembled, encoding='utf-8')
        if args.mesh_out:
            Path(args.mesh_out).write_text(payload['mesh_block'], encoding='utf-8')
        if not _write_output(payload, args.output):
            return 2
        return 2 if payload['validation']['status'] == 'failed' else 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        print('assist: recommend-mesh: ' + str(error), file=sys.stderr)
        return 2


def conditions_deck(c):
    """Small validation deck; material tables remain the solver's responsibility.

    Void layers and the padding beyond the target are the deck's one void material.
    """
    layout = layer_layout(c)
    solid = [layer for layer in c['layers'] if not layer.get('void')]
    materials = sorted(set(layer['material'] for layer in solid))
    atoms = {layer['material']: (layer['A'], layer['Z']) for layer in solid}
    if any(atoms[layer['material']] != (layer['A'], layer['Z']) for layer in solid):
        raise ValueError('layers with the same material name must have identical A and Z')
    if any(name.upper() == 'VOID' for name in materials):
        raise ValueError('material layer name VOID is reserved for the void material')
    area = 4 * math.pi * layout[-1][1]**2 if c['geometry'] == 'spherical' else (2 * math.pi * layout[-1][1] if c['geometry'] == 'cylindrical' else 1)
    rows = c['pulse']['intensity_table']
    lines = ['from tenryu_namelist import *', 'from bisect import bisect_right',
             'Main(name="mesh_recommendation", dimension="1D_SPH", temperature_model="2T", t_end=' + repr(c['t_end_s']) + ')',
             'Mesh(nr=128, r_min=' + repr(c['r_min_cm']) + ', r_max=' + repr(c['r_max_cm']) + ', geometry_1d=' + repr(c['geometry']) + ')']
    names = []
    for index, name in enumerate(materials):
        a, z = atoms[name]
        var = 'material_' + str(index)
        names.append(var)
        lines.append(var + '=Material(name=' + repr(name) + ', A=' + repr(a) + ', Z=' + repr(z) + ', eos=dict(model="ideal_gas"))')
    lines.extend(['void=Material(name="VOID", A=1.0, Z=1.0, is_void=True)',
                  'Materials(materials=[' + ','.join(names + ['void']) + '])'])
    rho_expr = '1e-9'
    for lo, hi, rho in reversed(layout):
        rho_expr = repr(rho) + ' if r < ' + repr(hi) + ' else (' + rho_expr + ')'
    fractions = []
    for name in materials:
        intervals = ['(' + repr(lo) + ' <= r < ' + repr(hi) + ')' for (lo, hi, rho), layer in zip(layout, c['layers'])
                     if layer['material'] == name and not layer.get('void')]
        fractions.append(repr(name) + ': lambda r: float(' + ' or '.join(intervals) + ')')
    voids = ['(' + repr(lo) + ' <= r < ' + repr(hi) + ')' for (lo, hi, rho), layer in zip(layout, c['layers']) if layer.get('void')]
    fractions.append('"VOID": lambda r: float(' + ' or '.join(voids + ['r >= ' + repr(layout[-1][1])]) + ')')
    lines.extend(['Geometry(rho=lambda r: ' + rho_expr + ', Te=lambda r: 1.0, Ti=lambda r: 1.0, volfrac={' + ','.join(fractions) + '})',
                  'Radiation(enabled=False)', 'TIMES=' + repr([r[0] for r in rows]), 'POWER=' + repr([r[1] * area for r in rows]),
                  'def power(t):\n    if t < TIMES[0] or t > TIMES[-1]:\n        return 0.0\n    i = min(len(TIMES)-1, max(1, bisect_right(TIMES,t)))\n    return POWER[i-1] + (POWER[i]-POWER[i-1])*(t-TIMES[i-1])/(TIMES[i]-TIMES[i-1])',
                  'Laser(enabled=True, mode="radial_absorption_1d", wavelength_nm=' + repr(c['wavelength_nm']) + ', rays_per_beam=8000, beams=[LaserBeam(name="beam", direction=(0.0,0.0,-1.0), power=power, focus=(0.0,0.0,0.0), profile=dict(model="super_gaussian", w0_um=250.0, m=4))])',
                  'Numerics(hydro=dict(driver_full_step_retry_enabled=True), conduction=dict(enabled=True, solver="implicit"))'])
    return '\n'.join(lines) + '\n'
