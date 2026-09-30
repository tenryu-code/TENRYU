"""Deterministic campaign-derived initial 1D mesh recommendations (stdlib only)."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

from tools.validation import mesh_convergence_campaign as campaign

REFERENCE = Path(__file__).resolve().parent / 'data/mesh_convergence_reference.json'
INTERIOR = 2.0e-5
QUANTILE = 0.90
SHAPES = {'W1': 'square', 'W2': 'square', 'W3': 'square', 'W4': 'gaussian',
          'W5': 'foot_main', 'W6': 'picket', 'W7': 'ramp', 'W8': 'long_low'}
SHAPE_WAVE = {shape: wave for wave, shape in SHAPES.items()}
SHAPE_WAVE['square'] = 'W2'
METHOD = 'log-ridge-local-residual-v2'
# Ion mass unit and eV as the solver's requirement model uses them (core/constants.hpp).
ION_MASS_G = 1.6726219e-24
EV_TO_ERG = 1.6022e-12
# Pulse shapes: normalized L1 distance on the pulse window below which a pulse belongs to
# a campaign waveform class; the window holds all but this fraction of the pulse energy,
# split equally before and after it.
SHAPE_TOLERANCE = 0.10
PULSE_WINDOW_ENERGY_OUTSIDE = 1e-2
# The solver freezes each beam's power callable on the absolute times t_k = k h,
# h = 2^-40 s (doubled while more than 2^20 base intervals would be needed), and bisects an
# interval while its chord misses the midpoint value by more than 1e-6 of the local
# magnitude, at most 7 times (core/namelist/frozen_table.cpp).
FROZEN_TIME_BASE_STEP_S = 2.0**-40
FROZEN_TIME_MAX_REFINE = 7
FROZEN_TIME_REL_TOL = 1e-6
FROZEN_TIME_MAX_BASE_INTERVALS = 1 << 20
# Effective pulse duration covered by the campaign waveforms, with a factor 1.25 margin.
DURATION_COVERAGE_MARGIN = 1.25
# Normalized feature distance below which a query counts as a measured case (about a 6 %
# change of one feature): the measured a_conv is resolved only to its factor-2 ladder step.
NEAR_MEASURED_DISTANCE = 0.025
# Density [g/cc] of a void layer given without one, and of the void padding beyond the
# target in the emitted density regions.
VOID_DENSITY = 1e-9


def positive(value, name, allow_zero=False):
    if isinstance(value, bool):
        raise ValueError(name + ' must be a finite number')
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(name + ' must be finite and ' + ('nonnegative' if allow_zero else 'positive'))
    return value


def interp(x, xp, yp):
    if x < xp[0] or x > xp[-1]:
        return 0.0
    hi = bisect.bisect_right(xp, x)
    if hi == len(xp):
        return yp[-1]
    lo = max(0, hi - 1)
    return yp[lo] + (yp[hi] - yp[lo]) * (x - xp[lo]) / (xp[hi] - xp[lo])


def _table_function(rows):
    """Linear interpolation of [time, value] rows; zero outside the table."""
    xs = [row[0] for row in rows]
    ys = [row[1] for row in rows]
    def value(t):
        if t < xs[0] or t > xs[-1]:
            return 0.0
        hi = bisect.bisect_right(xs, t)
        if hi >= len(xs):
            return ys[-1]
        lo = hi - 1
        return ys[lo] if xs[hi] == xs[lo] else ys[lo] + (ys[hi] - ys[lo]) * (t - xs[lo]) / (xs[hi] - xs[lo])
    return value


def freeze_time_table(function, t_end):
    """(times, values) of `function` frozen as the solver freezes a beam power callable."""
    h = FROZEN_TIME_BASE_STEP_S
    while math.ceil(t_end / h) > FROZEN_TIME_MAX_BASE_INTERVALS:
        h *= 2.0
    n_intervals = max(math.ceil(t_end / h), 1)
    xs, ys = [0.0], [function(0.0)]
    def refine(a, fa, b, fb, level):
        if level >= FROZEN_TIME_MAX_REFINE:
            return
        m = 0.5 * (a + b)
        fm = function(m)
        if not abs(fm - 0.5 * (fa + fb)) > FROZEN_TIME_REL_TOL * max(abs(fa), abs(fb), abs(fm)):
            return
        refine(a, fa, m, fm, level + 1)
        xs.append(m)
        ys.append(fm)
        refine(m, fm, b, fb, level + 1)
    t_prev, f_prev = 0.0, ys[0]
    for k in range(1, n_intervals + 1):
        t_next = float(k) * h
        f_next = function(t_next)
        refine(t_prev, f_prev, t_next, f_next, 0)
        xs.append(t_next)
        ys.append(f_next)
        t_prev, f_prev = t_next, f_next
    return xs, ys


def frozen_value(xs, ys, t):
    """Linear interpolation of a frozen table, zero outside it (FrozenTable1D::eval)."""
    if t < xs[0] or t > xs[-1]:
        return 0.0
    hi = bisect.bisect_right(xs, t)
    if hi >= len(xs):
        return ys[-1]
    lo = hi - 1
    dx = xs[hi] - xs[lo]
    if dx <= sys.float_info.epsilon:
        return ys[lo]
    return ys[lo] + (t - xs[lo]) / dx * (ys[hi] - ys[lo])


def _pulse_window(function, t_end, samples=20001):
    """Times before and after which PULSE_WINDOW_ENERGY_OUTSIDE / 2 of the energy on
    [0, t_end] is delivered (trapezoidal cumulative integral on `samples` points).

    An energy window is insensitive to where a low tail is cut: a Gaussian truncated at
    1.8 % of its peak and the same Gaussian run past its tail share it.
    """
    times = [t_end * i / (samples - 1) for i in range(samples)]
    values = [function(t) for t in times]
    cumulative = [0.0]
    for i in range(1, samples):
        cumulative.append(cumulative[-1] + .5 * (values[i - 1] + values[i]) * (times[i] - times[i - 1]))
    total = cumulative[-1]
    if not total > 0:
        return 0.0, t_end
    def time_at(target):
        j = bisect.bisect_left(cumulative, target)
        if j <= 0:
            return times[0]
        if j >= samples:
            return times[-1]
        span = cumulative[j] - cumulative[j - 1]
        fraction = (target - cumulative[j - 1]) / span if span > 0 else 0.0
        return times[j - 1] + fraction * (times[j] - times[j - 1])
    lo = time_at(.5 * PULSE_WINDOW_ENERGY_OUTSIDE * total)
    hi = time_at((1 - .5 * PULSE_WINDOW_ENERGY_OUTSIDE) * total)
    return (lo, hi) if hi > lo else (0.0, t_end)


def classify_pulse_shape(rows, peak, t_end, samples=4001):
    """Nearest campaign waveform class by the normalized L1 distance on each pulse window.

    Both pulses are normalized by their peak and compared in time relative to their own
    energy window, so a run that continues after the pulse, a pulse end that falls exactly
    on t_end (a discontinuity sampled one way or the other) and where a low tail is cut do
    not change the class; narrow features still count with their weight in the integral.
    """
    pulse = _table_function(sorted([float(t), float(v)] for t, v in rows))
    lo, hi = _pulse_window(pulse, t_end)
    best = None
    for wave in ('W2', 'W4', 'W5', 'W6', 'W7', 'W8'):
        definition = campaign.WAVEFORMS[wave]
        reference = lambda t, wave=wave: campaign.waveform_intensity(wave, t)
        wlo, whi = _pulse_window(reference, definition['t_end_s'])
        difference = norm = 0.0
        for i in range(samples):
            u = i / (samples - 1)
            weight = .5 if i in (0, samples - 1) else 1.0
            g = reference(wlo + u * (whi - wlo)) / definition['peak_W_cm2']
            f = pulse(lo + u * (hi - lo)) / peak
            difference += weight * abs(f - g)
            norm += weight * g
        distance = difference / norm
        if best is None or distance < best[0]:
            best = (distance, SHAPES[wave])
    return best


def pulse_effective_duration(rows, peak, t_end):
    """Integral of the intensity over [0, t_end] divided by the peak [s]."""
    rows = sorted([float(t), float(v)] for t, v in rows)
    points = [row for row in rows if row[0] <= t_end]
    if not points or points[-1][0] < t_end:
        points.append([t_end, _table_function(rows)(t_end)])
    if points[0][0] > 0:
        points.insert(0, [0.0, 0.0])
    return sum((b[0] - a[0]) * .5 * (a[1] + b[1]) for a, b in zip(points, points[1:])) / peak


def target_layers(c):
    """The material layers of normalized conditions: every layer but the void ones."""
    return [layer for layer in c['layers'] if not layer.get('void')]


def normalize_conditions(source):
    """Canonical, numeric conditions. Layers run from inner to laser-facing outer side.

    A layer with "void": true is vacuum inside the target (behind a planar foil with a
    free rear surface, inside a hollow shell, between shells): the mesh zones it as a void
    segment, and the target physics (features, coverage, ablation) sees only the material
    layers. It needs no A and Z; its density (the zoning weight of the void cells) defaults
    to VOID_DENSITY, which also replaces a zero void density (Materials.void_config.rho
    may be zero).
    """
    if not isinstance(source, dict):
        raise ValueError('conditions must be a JSON object')
    geometry = source.get('geometry', 'planar')
    if geometry not in ('planar', 'spherical', 'cylindrical'):
        raise ValueError('geometry must be planar, spherical or cylindrical')
    t_end = positive(source['t_end_s'], 't_end_s')
    wavelength = positive(source['wavelength_nm'], 'wavelength_nm')
    layers = []
    for layer in source['layers']:
        if not isinstance(layer, dict):
            raise ValueError('each layer must be an object')
        material = str(layer.get('material', '')).strip()
        if not material:
            raise ValueError('each layer needs material')
        void = layer.get('void', False)
        if not isinstance(void, bool):
            raise ValueError('layer void must be true or false (' + material + ')')
        if void:
            rho = positive(layer.get('rho_gcc', layer.get('rho', VOID_DENSITY)), 'void rho_gcc', True)
            layers.append(dict(material=material, void=True, rho_gcc=rho if rho > 0 else VOID_DENSITY,
                               thickness_cm=positive(layer['thickness_cm'], 'thickness_cm'), A=1.0, Z=0.0))
            continue
        if 'A' not in layer or 'Z' not in layer:
            raise ValueError('each layer needs explicit material A and Z (missing for ' + material + ')')
        layers.append(dict(material=material, rho_gcc=positive(layer.get('rho_gcc', layer.get('rho')), 'rho_gcc'),
                           thickness_cm=positive(layer['thickness_cm'], 'thickness_cm'),
                           A=positive(layer['A'], 'material A'), Z=positive(layer['Z'], 'material Z')))
    if not layers:
        raise ValueError('layers must not be empty')
    if layers[-1].get('void'):
        raise ValueError('the outermost layer must be material; the void beyond the target is the padding to r_max_cm')
    r_min = positive(source.get('r_min_cm', 0), 'r_min_cm', True)
    outer = r_min + sum(layer['thickness_cm'] for layer in layers)
    r_max = positive(source.get('r_max_cm', outer), 'r_max_cm')
    if r_max < outer * (1 - 1e-12):
        raise ValueError('r_max_cm must cover the layers')
    pulse = source['pulse']
    allowed = {'shape', 'duration_s', 'rise_s', 'fwhm_s', 'center_s',
               'peak_intensity_W_cm2', 'energy_J', 'area_cm2', 'spot_radius_cm',
               'intensity_table', 'power_table'}
    if set(pulse) - allowed:
        raise ValueError('unknown pulse keys: ' + ', '.join(sorted(set(pulse) - allowed)))
    if 'intensity_table' in pulse and 'power_table' in pulse:
        raise ValueError('specify intensity_table or power_table, not both')
    times = [t_end * i / 100 for i in range(101)]
    # The solver sees each pulse as its frozen table (freeze_time_table); every quantity
    # below reads that table, so a table exported by the solver and the conditions it was
    # made from give the same numbers even where the pulse jumps.
    if 'intensity_table' in pulse or 'power_table' in pulse:
        is_power = 'power_table' in pulse
        rows = pulse['power_table' if is_power else 'intensity_table']
        if not isinstance(rows, list) or len(rows) < 2 or any(not isinstance(row, (list, tuple)) or len(row) != 2 for row in rows):
            raise ValueError('pulse table needs at least two [time_s, value] rows')
        xp = [positive(row[0], 'pulse time', True) for row in rows]
        yp = [positive(row[1], 'pulse value', True) for row in rows]
        if any(b <= a for a, b in zip(xp, xp[1:])):
            raise ValueError('pulse table times must strictly increase')
        if is_power:
            area = pulse.get('area_cm2')
            if area is None:
                area = 4 * math.pi * outer**2 if geometry == 'spherical' else (
                    2 * math.pi * outer if geometry == 'cylindrical' else 1.0)
            area = positive(area, 'area_cm2')
            yp = [y / area for y in yp]
        drive = lambda t: interp(t, xp, yp)
    else:
        shape = pulse.get('shape')
        if shape not in SHAPE_WAVE:
            raise ValueError('pulse.shape must be ' + ', '.join(SHAPE_WAVE))
        wave = SHAPE_WAVE[shape]
        definition = campaign.WAVEFORMS[wave]
        duration = positive(pulse.get('duration_s', t_end), 'duration_s')
        base_duration = definition['t_end_s']
        def unit_value(t):
            if shape in ('square', 'long_low'):
                rise = positive(pulse.get('rise_s', .1e-9), 'rise_s', True)
                return min(1.0, t / rise) if rise and 0 <= t <= duration else float(0 <= t <= duration)
            if shape == 'gaussian':
                fwhm = positive(pulse.get('fwhm_s', duration / 2.4), 'fwhm_s')
                center = positive(pulse.get('center_s', duration / 2), 'center_s', True)
                return math.exp(-4 * math.log(2) * ((t - center) / fwhm)**2) if t <= duration else 0
            return campaign.waveform_intensity(wave, t * base_duration / duration) / definition['peak_W_cm2']
        unit = [unit_value(t) for t in times]
        if 'peak_intensity_W_cm2' in pulse:
            peak = positive(pulse['peak_intensity_W_cm2'], 'peak_intensity_W_cm2')
        else:
            energy = positive(pulse['energy_J'], 'energy_J')
            area = positive(pulse.get('area_cm2', math.pi * positive(pulse.get('spot_radius_cm', 0), 'spot_radius_cm')**2)
                            if 'area_cm2' not in pulse else pulse['area_cm2'], 'area_cm2')
            integral = sum((a + b) * .5 * t_end / 100 for a, b in zip(unit, unit[1:]))
            peak = energy / area / positive(integral, 'pulse integral')
        drive = lambda t: unit_value(t) * peak
    xs, ys = freeze_time_table(drive, t_end)
    canonical_rows = [[x, y] for x, y in zip(xs, ys)]
    values = [frozen_value(xs, ys, t) for t in times]
    peak = max([y for x, y in canonical_rows if x <= t_end] + [frozen_value(xs, ys, t_end)])
    if not peak > 0 or max(values) <= 0:
        raise ValueError('no positive drive during the run')
    peak = positive(peak, 'pulse peak')
    shape_error, shape = classify_pulse_shape(canonical_rows, peak, t_end)
    effective = pulse_effective_duration(canonical_rows, peak, t_end)
    physics = source.get('physics', {'eos': 'tmat', 'radiation_enabled': False,
                                  'temperature_model': '2T', 'conduction_solver': 'implicit'})
    return dict(wavelength_nm=wavelength, geometry=geometry, layers=layers,
                t_end_s=t_end, r_min_cm=r_min, r_max_cm=r_max,
                pulse=dict(intensity_table=canonical_rows), sampled_intensity_W_cm2=values,
                peak_intensity_W_cm2=peak, shape=shape, shape_error=shape_error,
                effective_duration_s=effective, physics=physics)


def reference_conditions(row):
    wave = campaign.WAVEFORMS[row['waveform']]
    duration = wave['t_end_s']
    layers = [dict(material='CD', A=7.0, Z=3.5, rho_gcc=l['rho'], thickness_cm=l['thickness_cm']) for l in row['layers']]
    outer = sum(l['thickness_cm'] for l in layers)
    return normalize_conditions(dict(wavelength_nm=row['wavelength_nm'], geometry=row['geometry'],
        layers=layers, t_end_s=duration,
        r_max_cm=campaign.SHELL_R_MAX if row['geometry'] == 'spherical' else outer + campaign.VOID_THICKNESS,
        pulse=dict(intensity_table=[[duration * i / 9999, campaign.waveform_intensity(row['waveform'], duration * i / 9999)] for i in range(10000)])))


def features(c):
    layers = target_layers(c)
    pulse = [value / c['peak_intensity_W_cm2'] for value in c['sampled_intensity_W_cm2']]
    effective = c['t_end_s'] * sum((a + b) * .5 for a, b in zip(pulse, pulse[1:])) / 100
    crossing = next((i for i, v in enumerate(pulse) if v >= .9), 100)
    rise = max(c['t_end_s'] * crossing / 100, c['t_end_s'] / 100)
    target = layers[-1:] if c['geometry'] == 'spherical' else layers
    return [math.log10(c['wavelength_nm'] / 351), math.log10(c['peak_intensity_W_cm2'] / 1e14),
            math.log10(layers[-1]['rho_gcc']), math.log10(sum(l['thickness_cm'] for l in target) / .0025),
            math.log10(effective / 1e-9), math.log10(rise / 1e-10),
            float(c['geometry'] == 'spherical'), float(len(layers) > 1),
            math.log10(layers[0]['rho_gcc'] / layers[-1]['rho_gcc'])]


def same_conditions(a, b):
    if a['geometry'] != b['geometry'] or len(a['layers']) != len(b['layers']):
        return False
    pairs = [(a[k], b[k]) for k in ('wavelength_nm', 't_end_s', 'r_min_cm', 'peak_intensity_W_cm2')]
    for la, lb in zip(a['layers'], b['layers']):
        if la['material'].upper() != lb['material'].upper() or bool(la.get('void')) != bool(lb.get('void')):
            return False
        pairs.extend((la[k], lb[k]) for k in ('rho_gcc', 'thickness_cm', 'A', 'Z'))
    if any(abs(x - y) > 2e-4 * max(abs(x), abs(y), 1e-20) for x, y in pairs):
        return False
    peak = max(a['peak_intensity_W_cm2'], b['peak_intensity_W_cm2'])
    return max(abs(x - y) / peak for x, y in zip(a['sampled_intensity_W_cm2'], b['sampled_intensity_W_cm2'])) <= 2e-3


def solve(matrix, vector):
    """Small pivoted dense solve; ridge keeps the feature system nonsingular."""
    rows = [list(row) + [value] for row, value in zip(matrix, vector)]
    n = len(rows)
    for i in range(n):
        pivot = max(range(i, n), key=lambda j: abs(rows[j][i]))
        rows[i], rows[pivot] = rows[pivot], rows[i]
        scale = rows[i][i]
        rows[i] = [v / scale for v in rows[i]]
        for j in range(n):
            if j != i:
                scale = rows[j][i]
                rows[j] = [a - scale * b for a, b in zip(rows[j], rows[i])]
    return [row[-1] for row in rows]


class CampaignModel:
    def __init__(self, path=REFERENCE):
        data = Path(path).read_bytes()
        self.digest = hashlib.sha256(data).hexdigest()
        payload = json.loads(data)
        if payload['schema'] != 'tenryu.mesh_convergence_reference.v1':
            raise ValueError('unsupported convergence reference schema')
        self.rows = [row for row in payload['cases'] if '-S' not in row['id']]
        self.conditions = {row['id']: reference_conditions(row) for row in self.rows}
        self.converged = [row for row in self.rows if row['converged'] and row['a_conv_g_cm2'] is not None]
        durations = [self.conditions[row['id']]['effective_duration_s'] for row in self.rows]
        self.duration_range = (min(durations), max(durations))
        self.cap = max(row['a_conv_g_cm2'] / self.reference_apriori(row) for row in self.converged)
        self.evidence_radius = self.spacing_radius(self.converged)
        loo = []
        for row in self.converged:
            training = [r for r in self.converged if r['id'] != row['id']]
            raw, evidence = self.predict(self.conditions[row['id']], training)
            loo.append(dict(id=row['id'], measured_g_cm2=row['a_conv_g_cm2'], raw_g_cm2=raw,
                            error_dex=math.log10(raw / row['a_conv_g_cm2']),
                            apriori_g_cm2=self.reference_apriori(row),
                            **self.relaxation_evidence(evidence, training)))
        errors = sorted(item['error_dex'] for item in loo)
        self.margin = max(0, errors[math.ceil(QUANTILE * len(errors)) - 1]) + 1e-12
        for item in loo:
            item['adjusted_g_cm2'] = item['raw_g_cm2'] * 10**(-self.margin)
            item['adjusted_safe'] = item['adjusted_g_cm2'] <= item['measured_g_cm2']
            item['final_g_cm2'] = min(item['adjusted_g_cm2'], item['allowed_apriori_factor'] * item['apriori_g_cm2'])
            item['safe'] = item['final_g_cm2'] <= item['measured_g_cm2']
            item['unsafe_loosening'] = not item['safe'] and item['final_g_cm2'] > item['apriori_g_cm2']
        self.loo = dict(quantile=QUANTILE, margin_dex=self.margin, n_cases=len(loo),
                        n_adjusted_safe=sum(r['adjusted_safe'] for r in loo),
                        n_safe=sum(r['safe'] for r in loo),
                        n_unsafe_loosening=sum(r['unsafe_loosening'] for r in loo),
                        evidence_radius=self.evidence_radius,
                        safe_fraction=sum(r['safe'] for r in loo) / len(loo), cases=loo)

    def spacing_radius(self, rows):
        """Median nearest-neighbor spacing, determined from training inputs only."""
        points = [features(self.conditions[r['id']]) for r in rows]
        return statistics.median(min(math.sqrt(sum((a - b)**2 for a, b in zip(x, y)))
            for j, y in enumerate(points) if i != j) for i, x in enumerate(points))

    def relaxation_evidence(self, evidence, rows):
        radius = self.spacing_radius(rows)
        distance = evidence[0]['distance']
        ratios = {r['id']: r['a_conv_g_cm2'] / self.reference_apriori(r) for r in rows}
        ratio = sum(e['weight'] * ratios[e['id']] for e in evidence)
        proximity = max(0.0, 1.0 - distance / radius) if radius > 0 else 0.0
        return dict(evidence_distance=distance, evidence_radius=radius,
                    neighbor_apriori_ratio=ratio,
                    allowed_apriori_factor=min(self.cap, 1 + proximity * max(0.0, ratio - 1)))

    @staticmethod
    def reference_apriori(row):
        peak = campaign.WAVEFORMS[row['waveform']]['peak_W_cm2']
        return row['ceiling_formation_g_cm2'] * (8 / 9) * min(1, (peak / 1e14)**-.4)

    def predict(self, conditions, rows=None):
        rows = self.converged if rows is None else rows
        x = [[1.0] + features(self.conditions[r['id']]) for r in rows]
        y = [math.log10(r['a_conv_g_cm2']) for r in rows]
        n = len(x[0])
        matrix = [[sum(v[i] * v[j] for v in x) + (0.5 if i == j and i else 0) for j in range(n)] for i in range(n)]
        beta = solve(matrix, [sum(v[i] * target for v, target in zip(x, y)) for i in range(n)])
        query = [1.0] + features(conditions)
        trend = sum(a * b for a, b in zip(query, beta))
        neighbors = []
        for row, point, target in zip(rows, x, y):
            distance = math.sqrt(sum((a - b)**2 for a, b in zip(query, point)))
            residual = target - sum(a * b for a, b in zip(point, beta))
            neighbors.append((distance, row['id'], residual, target))
        neighbors = sorted(neighbors)[:5]
        weights = [1 / max(d, .05)**2 for d, _, _, _ in neighbors]
        total = sum(weights)
        evidence = [dict(id=identifier, distance=distance, weight=w / total, measured_g_cm2=10**target)
                    for (distance, identifier, residual, target), w in zip(neighbors, weights)]
        prediction = trend + sum(w / total * row[2] for row, w in zip(neighbors, weights))
        return 10**prediction, evidence

    def coverage_reasons(self, c):
        reasons = []
        layers = target_layers(c)
        if len(layers) != len(c['layers']):
            reasons.append('void inside the target (the campaign targets have a fixed rear wall or a gas fill)')
        if any(l['material'].upper() != 'CD' for l in layers):
            reasons.append('material outside CD')
        if any(abs(l['A'] / 7 - 1) > 2e-4 or abs(l['Z'] / 3.5 - 1) > 2e-4 for l in layers):
            reasons.append('material A/Z outside campaign CD average atom (A=7, Z=3.5)')
        if c['wavelength_nm'] not in (351, 527, 1053):
            reasons.append('wavelength outside measured 351/527/1053 nm')
        if not 3e13 * (1 - 1e-6) <= c['peak_intensity_W_cm2'] <= 1e15 * (1 + 1e-6):
            reasons.append('intensity outside 3e13–1e15 W/cm2')
        low, high = self.duration_range
        if not low / DURATION_COVERAGE_MARGIN <= c['effective_duration_s'] <= high * DURATION_COVERAGE_MARGIN:
            reasons.append('effective pulse duration outside the campaign waveforms ({0:.2g}–{1:.2g} ns)'.format(
                1e9 * low / DURATION_COVERAGE_MARGIN, 1e9 * high * DURATION_COVERAGE_MARGIN))
        if c['shape_error'] > SHAPE_TOLERANCE:
            reasons.append('pulse shape outside sampled campaign waveforms')
        if c['geometry'] == 'cylindrical':
            reasons.append('cylindrical geometry has no measured cases')
        if c['geometry'] == 'spherical':
            shell = self.conditions['C28']
            if len(layers) != 2 or any(abs(a[k] / b[k] - 1) > 2e-4
                    for a, b in zip(layers, shell['layers']) for k in ('rho_gcc', 'thickness_cm')):
                reasons.append('shell radius, thickness or fill outside GXII cases')
        else:
            if any(not .05 <= l['rho_gcc'] <= 2.5 for l in layers):
                reasons.append('density outside 0.05–2.5 g/cc')
            thickness = sum(l['thickness_cm'] for l in layers)
            if not .001 * (1 - 1e-6) <= thickness <= .052 * (1 + 1e-6) or len(layers) > 2:
                reasons.append('layering or thickness outside measured foils')
        if c['r_min_cm'] != 0:
            reasons.append('nonzero inner radius outside campaign')
        if c['physics'] != dict(eos='tmat', radiation_enabled=False, temperature_model='2T', conduction_solver='implicit'):
            reasons.append('physics differs from radiation-off CD tmat, 2T implicit conduction')
        return reasons

    def recommend(self, source, requirement=None, *, exclude_case_id=None):
        """Recommend; exclude_case_id supports the held-out audit without exact retrieval."""
        c = normalize_conditions(source)
        reasons = self.coverage_reasons(c)
        flags, warnings = [], []
        if len(target_layers(c)) != len(c['layers']):
            # The 1D hydro re-spaces every run of void cells evenly between the faces around it
            # (main 125a94572) and has no contact model: a void that closes stops the run.
            flags.append('interior_void')
            warnings.append('Void inside the target: the 1D solver keeps the void cells evenly spaced between '
                            'the faces around them (main 125a94572) but has no contact model, so the run stops '
                            'if the void closes. Make it wider than the travel of those faces during the run '
                            '(a 25 um CD foil at 1e14 W/cm2 moved its rear face about 40 um in 1 ns).')
        rows = [r for r in self.rows if r['id'] != exclude_case_id]
        training = [r for r in self.converged if r['id'] != exclude_case_id]
        raw, evidence = self.predict(c, training)
        gate = self.relaxation_evidence(evidence, training)
        adjusted = raw * 10**(-self.margin)
        exact = [row for row in rows if same_conditions(c, self.conditions[row['id']])]
        near = None
        if not exact and evidence and evidence[0]['distance'] <= NEAR_MEASURED_DISTANCE:
            nearest = next(row for row in rows if row['id'] == evidence[0]['id'])
            reference = self.conditions[nearest['id']]
            if (nearest['converged'] and reference['geometry'] == c['geometry'] and
                    len(reference['layers']) == len(c['layers']) and reference['shape'] == c['shape']):
                near = nearest
        integral = apriori_integral(c)
        apriori = integral['ceiling_formation_g_cm2']
        mu_abl = integral['mu_abl_total_g_cm2']
        mu_source = 'Python formation integral'
        apriori_source = 'material-aware calibrated formation integral estimate'
        if exact and not reasons:
            apriori = self.reference_apriori(exact[0])
            apriori_source = 'calibrated shipped reference ceiling'
        if requirement and requirement.get('applicable'):
            ablation = requirement['ablation']
            apriori = ablation.get('apriori_ceiling_formation_g_cm2', ablation['ceiling_formation_g_cm2'])
            apriori = positive(apriori, 'solver apriori ceiling')
            apriori_source = 'solver mesh_requirement'
            if 'mu_abl_total_g_cm2' in ablation:
                mu_abl = positive(ablation['mu_abl_total_g_cm2'], 'solver ablated depth')
                mu_source = 'solver mesh_requirement'
        unconverged = [r for r in rows if not r['converged'] and (
            c['wavelength_nm'] == 1053 and r['wavelength_nm'] == 1053 or
            c['wavelength_nm'] == 527 and c['peak_intensity_W_cm2'] >= 1e15 * (1 - 1e-6) and r['id'] == 'C06' or
            c['geometry'] == 'planar' and low_density_ablated(c, mu_abl) and r['id'] == 'C15')]
        if reasons:
            flags.append('extrapolation')
            warnings.extend(reasons)
            # No empirical relaxation outside coverage, even when a failed class also matches.
            adjusted = apriori
            gate['allowed_apriori_factor'] = 1.0
            mode, confidence = 'apriori_fallback', 'outside campaign; convergence pair required'
        elif (exact and exact[0]['converged']) or near is not None:
            matched = exact[0] if exact else near
            distance = 0.0 if exact else evidence[0]['distance']
            raw = matched['a_conv_g_cm2']
            adjusted = .95 * raw
            evidence = [dict(id=matched['id'], distance=distance, weight=1.0, measured_g_cm2=raw)]
            gate.update(evidence_distance=distance, neighbor_apriori_ratio=raw / apriori,
                        allowed_apriori_factor=min(self.cap, adjusted / apriori))
            mode = 'measured_case'
            confidence = ('measured case; 5% zoning headroom, campaign observables only' if exact else
                          'near a measured case (feature distance {0:.3g}); 5% zoning headroom'.format(distance))
        else:
            mode, confidence = 'local_trend', 'interpolation; LOO calibration coverage is not a convergence guarantee'
        margin_adjusted = adjusted
        if unconverged:
            flags.append('unconverged_reference')
            confidence = 'unconverged reference class; even the finest run is not proven converged'
            finest = min(campaign.A_SURF_LADDER[max(int(k) for k in r['n_cells_per_level'])] for r in unconverged)
            if not reasons:
                adjusted = min(adjusted, finest)
            warnings.append('Unconverged reference class: mesh is not proven converged. ' + (
                'Outside coverage retain the material a-priori fallback and run a convergence pair.' if reasons else
                'Use finest run level or finer.'))
        else:
            finest = None
        surface = min(adjusted, (self.cap if mode == 'measured_case' else gate['allowed_apriori_factor']) * apriori)
        if mode == 'local_trend' and surface < min(adjusted, self.cap * apriori):
            flags.append('sparse_evidence')
            warnings.append('Nearby evidence limits loosening of the calibrated a-priori ceiling; inspect evidence_distance and allowed_apriori_factor.')
        empirical = None
        if not reasons:
            empirical = dict(reference_sha256=self.digest, case_ids=[item['id'] for item in evidence],
                             surface_ceiling_g_cm2=surface, reference_apriori_g_cm2=apriori)
        return dict(schema='tenryu.assist.mesh_recommendation.v1', method=METHOD, conditions=c,
                    recommendation=dict(surface_areal_mass_g_cm2=surface, interior_areal_mass_g_cm2=INTERIOR,
                                        raw_g_cm2=raw, margin_adjusted_g_cm2=margin_adjusted, mode=mode,
                                        apriori_g_cm2=apriori, apriori_source=apriori_source,
                                        mu_abl_total_g_cm2=mu_abl, mu_abl_source=mu_source,
                                        apriori_factor=surface / apriori, relaxation_cap=self.cap, empirical=empirical, **gate),
                    evidence=evidence, unconverged_case_ids=[r['id'] for r in unconverged], finest_run_g_cm2=finest,
                    reference_sha256=self.digest, loo=self.loo, flags=flags, confidence=confidence, warnings=warnings)


def low_density_ablated(c, mu_abl, threshold=.05):
    """Whether the predicted ablated depth reaches a layer at or below `threshold` g/cc.

    The campaign's unconverged low-density class (C15) is an ablation front running
    through 0.05 g/cc foam; a light layer the front never reaches (a gas behind a pusher)
    is not in it, and neither is a void layer.
    """
    depth = 0.0
    for layer in reversed(target_layers(c)):
        if layer['rho_gcc'] <= threshold and depth < mu_abl:
            return True
        depth += layer['rho_gcc'] * layer['thickness_cm']
    return False


def zbar_tf(rho, temperature, Z, A):
    """Thomas-Fermi mean ionization, More (1985) Table IV, as src/materials/zbar_math.hpp."""
    Z = max(Z, 0.0)
    A = max(A, 1.0e-30)
    if not Z > 0.0:
        return 0.0
    rho = rho if 0.0 < rho < 1.0e300 else 1.0e-30
    temperature = temperature if 0.0 < temperature < 1.0e300 else 0.0
    R = max(rho / (Z * A), 1.0e-300)
    T0 = temperature / Z**(4.0 / 3.0)
    TF = T0 / (1.0 + T0)
    TF7 = TF**7
    A_fit = 0.003323 * T0**0.9718 + 9.26148e-5 * T0**3.10165
    B_fit = -math.exp(-1.7630 + 1.43175 * TF + 0.31546 * TF7)
    C_fit = -0.366667 * TF + 0.983333
    Q1 = A_fit * R**B_fit
    Q = (R**C_fit + Q1**C_fit)**(1.0 / C_fit)
    x = 14.3139 * Q**0.6624
    if not x < 1.0e300:
        return Z
    return min(Z, max(0.0, Z * x / (1.0 + x + math.sqrt(1.0 + 2.0 * x))))


def material_critical_density(A, Z, wavelength_nm, peak):
    """Default solver ionization rule; no material-name inference or Zbar override."""
    nc = 1.11485e21 / (wavelength_nm * 1e-3)**2
    zbar = Z
    if Z > 18:
        zbar = Z / 2
        for _ in range(5):
            zbar = min(Z, max(1e-3, zbar))
            rho_c = nc * ION_MASS_G * A / zbar
            speed = (peak * 1e7 / (4 * rho_c))**(1 / 3)
            temperature = A * ION_MASS_G * speed**2 / ((zbar + 1) * EV_TO_ERG)
            zbar = zbar_tf(rho_c, temperature, Z, A)
    return nc * ION_MASS_G * A / min(Z, max(1e-3, zbar))


def apriori_integral(c):
    """Solver formation integral on 10000 times, for exact piecewise-constant layers.

    Defaults: absorbed fraction 1, scale factor .12, mass safety 1.5,
    formation fraction .1, 9 zones, and the calibrated intensity correction. Void layers
    carry no depth, as the solver's non-void depth.
    """
    layers = c['layers']
    radius = c['r_min_cm'] + sum(l['thickness_cm'] for l in layers)
    power = {'planar': 1, 'cylindrical': 2, 'spherical': 3}[c['geometry']]
    position, depth = c['r_min_cm'], 0.0
    columns = []
    for layer in layers:
        end = position + layer['thickness_cm']
        if not layer.get('void'):
            column = layer['rho_gcc'] * (end**power - position**power) / (power * radius**(power - 1))
            critical = material_critical_density(layer['A'], layer['Z'], c['wavelength_nm'], c['peak_intensity_W_cm2'])
            columns.append((column, critical))
        position = end
    fronts = []
    for column, critical in reversed(columns):
        depth += column
        fronts.append((depth, critical))
    xp, yp = zip(*c['pulse']['intensity_table'])
    mass, length, previous_speed, previous_rate = [0.0], [0.0], 0.0, 0.0
    for i in range(10000):
        t = c['t_end_s'] * i / 9999
        rho_c = next((rho for limit, rho in fronts if mass[-1] <= limit), fronts[-1][1])
        speed = (interp(t, xp, yp) * 1e7 / (4 * rho_c))**(1 / 3)
        rate = rho_c * speed
        if i:
            dt = t - c['t_end_s'] * (i - 1) / 9999
            mass.append(mass[-1] + 1.5 * .5 * (previous_rate + rate) * dt)
            scale = length[-1] + .12 * .5 * (previous_speed + speed) * dt
            length.append(min(scale, radius / 2) if c['geometry'] == 'spherical' else scale)
        previous_speed, previous_rate = speed, rate
    formation = .1 * mass[-1]
    hi = bisect.bisect_left(mass, formation)
    if hi == 0:
        raise ValueError('apriori estimate needs positive integrated laser drive')
    fraction = (formation - mass[hi - 1]) / (mass[hi] - mass[hi - 1])
    formation_length = length[hi - 1] + fraction * (length[hi] - length[hi - 1])
    return dict(ceiling_formation_g_cm2=positive(columns[-1][1] * formation_length / 9 * min(1, (c['peak_intensity_W_cm2'] / 1e14)**-.4), 'apriori estimate'),
                mu_abl_total_g_cm2=mass[-1])


def apriori_estimate(c):
    return apriori_integral(c)['ceiling_formation_g_cm2']
