"""Record Python namelists, recover form inputs and verify import fidelity.

Public operations always execute user code in a timed child process. No solver,
third-party packages, assistant configuration, or server profile is needed.
"""

import ast
import inspect
import json
import linecache
import math
import os
import re
import reprlib
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import traceback

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.assist.deck_import_runtime import _studio_capture
from tools.assist.deck_import_mesh import mesh_points, sampling_note


def close(a, b, tolerance=1e-12):
    if hasattr(a, "tolist") or hasattr(b, "tolist"):
        return type(a) is type(b) and close(a.tolist(), b.tolist(), tolerance)
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance * max(abs(a), abs(b), 1e-300)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k], tolerance) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(close(x, y, tolerance) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def numeric_hints(fn):
    """Candidate knots from constants, closed-over values and safe arithmetic.

    Hints only propose points. No family is accepted without an independent
    dense-grid comparison including both neighbours of every proposed knot.
    """
    values = set()

    def visit(value, depth=0):
        if depth > 3:
            return
        if type(value) in (int, float) and math.isfinite(value):
            values.add(float(value))
        elif isinstance(value, (list, tuple)) and len(value) < 10000:
            for item in value:
                visit(item, depth + 1)
        elif isinstance(value, dict) and len(value) < 1000:
            for item in value.values():
                visit(item, depth + 1)

    env = dict(getattr(fn, "__globals__", {}))
    try:
        closure = inspect.getclosurevars(fn)
        env.update(closure.nonlocals)
    except TypeError:
        pass
    for key, value in env.items():
        if not key.startswith("__"):
            visit(value)
    visit(getattr(fn, "__defaults__", ()))
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        permitted = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.BinOp,
                     ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
                     ast.USub, ast.UAdd, ast.List, ast.Tuple)
        for node in ast.walk(tree):
            if isinstance(node, ast.expr) and all(isinstance(n, permitted) for n in ast.walk(node)):
                try:
                    # Names must be plain numbers/containers, never custom objects.
                    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
                    if any(type(env.get(n)) not in (int, float, list, tuple) for n in names):
                        continue
                    visit(eval(compile(ast.Expression(node), "<knot>", "eval"), {"__builtins__": {}}, env))
                except (ArithmeticError, TypeError, ValueError, NameError):
                    pass
    except (OSError, TypeError, SyntaxError, IndentationError):
        pass
    return sorted(values)


def grid(lo, hi, hints=(), count=2048):
    points = {lo + (hi - lo) * i / count for i in range(count + 1)}
    for x in hints:
        if lo <= x <= hi:
            points.add(x)
            points.add(max(lo, math.nextafter(x, -math.inf)))
            points.add(min(hi, math.nextafter(x, math.inf)))
    return sorted(points)


def finite_value(fn, *args):
    value = fn(*args) if callable(fn) else fn
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (tuple, list)):
        return [finite_value(v) for v in value]
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Profile returned a non-finite or non-numeric value: " + repr(value))
    return value


def corona_profile(blocks):
    """Recognize the generator's exponential corona without its state header."""
    fn = blocks.get("Geometry", {}).get("rho")
    if not callable(fn) or blocks.get("Main", {}).get("dimension") == "2D_RZ":
        return None
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        def number(node):
            allowed = (ast.Constant, ast.UnaryOp, ast.USub, ast.UAdd)
            if not all(isinstance(n, allowed) for n in ast.walk(node)):
                raise ValueError("Not a literal")
            return float(ast.literal_eval(node))
        for branch in ast.walk(tree):
            if not isinstance(branch, ast.If) or len(branch.body) != 1 or not isinstance(branch.body[0], ast.Return):
                continue
            expr = branch.body[0].value
            if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name) or expr.func.id != "max" or len(expr.args) != 2:
                continue
            product, floor = expr.args
            exponent = product.right.args[0]
            start = number(exponent.left.operand.right)
            scale = number(exponent.right)
            end = number(branch.test.comparators[0])
            peak = number(product.left)
            minimum = number(floor)
            if not (scale > 0 and end > start and peak > minimum > 0):
                continue
            return {"start": start, "scaleUm": scale*1e4, "extentUm": (end-start)*1e4, "rho0": peak, "rhoMin": minimum}
    except (OSError, TypeError, SyntaxError, AttributeError, ValueError, IndexError):
        pass
    return None


def curve(fn, end):
    """Recognize constants, Gaussian pulses and verified linear tables."""
    if not callable(fn):
        return {"kind": "constant", "value": fn}
    # Explicit time tables may extend beyond t_end. Recover their last knot
    # instead of losing editable input that is present in the source.
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        env = getattr(fn, "__globals__", {})
        for node in ast.walk(tree):
            if isinstance(node, (ast.List, ast.Tuple)):
                allowed = (ast.List, ast.Tuple, ast.Constant, ast.Name, ast.Load, ast.BinOp,
                           ast.Add, ast.Sub, ast.Mult, ast.Div, ast.UnaryOp, ast.USub, ast.UAdd)
                if all(isinstance(n, allowed) for n in ast.walk(node)) and all(type(env.get(n.id)) in (int,float) for n in ast.walk(node) if isinstance(n,ast.Name)):
                    values = eval(compile(ast.Expression(node), "<table-knots>", "eval"), {"__builtins__":{}}, env)
                    if len(values) >= 2 and all(type(v) in (int,float) for v in values) and values[0] >= 0 and all(a<b for a,b in zip(values,values[1:])) and end < values[-1] <= 100*end:
                        end = values[-1]
    except (OSError, TypeError, SyntaxError, ValueError, ArithmeticError):
        pass
    hints = [x for x in numeric_hints(fn) if 0 <= x <= end]
    xs = grid(0, end, hints)
    ys = [finite_value(fn, x) for x in xs]
    scale = max(map(abs, ys), default=0)
    if all(close(y, ys[0]) for y in ys):
        return {"kind": "constant", "value": ys[0]}
    # Log-quadratic fit in normalized time avoids ill-conditioning at ns units.
    positive = [i for i, y in enumerate(ys) if y > scale * 1e-5]
    if len(positive) >= 3:
        ids = [positive[0], positive[len(positive) // 2], positive[-1]]
        x0, x1, x2 = [xs[i] / end for i in ids]
        y0, y1, y2 = [math.log(ys[i]) for i in ids]
        a = ((y2-y1)/(x2-x1) - (y1-y0)/(x1-x0))/(x2-x0)
        b = (y1-y0)/(x1-x0) - a*(x0+x1)
        c = y0 - a*x0*x0 - b*x0
        if a < 0:
            center = -b/(2*a)
            log_peak = c - a*center*center
            if log_peak < 700:
                peak = math.exp(log_peak)
                width = math.sqrt(-4*math.log(2)/a)*end
                def gaussian(t):
                    return peak*math.exp(-4*math.log(2)*((t-center*end)/width)**2)
                if all(abs(gaussian(x)-y) <= 2e-13*scale for x, y in zip(xs, ys)):
                    return {"kind": "gaussian", "peak": peak, "center": center*end, "fwhm": width}
    # Locate jumps and slope changes even when their knots are not literals.
    coarse = grid(0, end, count=2048)
    cy = [finite_value(fn, x) for x in coarse]
    knots = set([0.0, end] + hints)
    for i in range(1, len(coarse)-1):
        left = (cy[i]-cy[i-1])/(coarse[i]-coarse[i-1])
        right = (cy[i+1]-cy[i])/(coarse[i+1]-coarse[i])
        if abs(left-right)*end <= 1e-9*max(scale, 1e-300):
            continue
        lo, hi = coarse[i-1], coarse[i+1]
        ylo, yhi = finite_value(fn, lo), finite_value(fn, hi)
        # A discontinuity between plateaux is bisected to adjacent floats.
        if i >= 2 and i+2 < len(coarse) and cy[i-2] == ylo and cy[i+2] == yhi:
            for _ in range(80):
                mid = (lo+hi)/2
                if mid == lo or mid == hi:
                    break
                if finite_value(fn, mid) == ylo:
                    lo = mid
                else:
                    hi = mid
            knots.update((lo, hi))
        # Intersect the lines on the two sides of a piecewise-linear corner.
        if i >= 2 and i+2 < len(coarse):
            sl = (cy[i-1]-cy[i-2])/(coarse[i-1]-coarse[i-2])
            sr = (cy[i+2]-cy[i+1])/(coarse[i+2]-coarse[i+1])
            if sl != sr:
                x = (cy[i+1]-sr*coarse[i+1]-cy[i-1]+sl*coarse[i-1])/(sl-sr)
                if coarse[i-1] <= x <= coarse[i+1]:
                    knots.add(x)
    # Square pulse convention is checked over the requested simulation time.
    if ys[0] > 0 and all(y == 0 or close(y, ys[0]) for y in ys):
        transitions = sorted(x for x in knots if 0 < x < end)
        for x in transitions:
            if all(close(y, ys[0] if t <= x else 0) for t, y in zip(xs + coarse, ys + cy)):
                return {"kind": "square", "power": ys[0], "duration": x}
    points = []
    for x in sorted(knots):
        y = finite_value(fn, x)
        while len(points) >= 2:
            xa, ya = points[-2]
            xb, yb = points[-1]
            interp = ya+(y-ya)*(xb-xa)/(x-xa)
            if abs(interp-yb) > 2e-14*max(scale, 1e-300):
                break
            points.pop()
        points.append([x, y])
    # Nearly coincident knots represent jumps, which the table editor cannot
    # express robustly. Keep the source instead of smoothing a discontinuity.
    if len(points) > 512 or any(b[0]-a[0] <= end*1e-13 for a, b in zip(points, points[1:])):
        return {"kind": "source", "reason": "Discontinuous or non-linear waveform"}
    def interpolate(t):
        import bisect
        j = min(len(points)-2, max(0, bisect.bisect_right([p[0] for p in points], t)-1))
        a, b = points[j], points[j+1]
        w = (t-a[0])/(b[0]-a[0])
        return a[1]*(1-w)+b[1]*w
    check = grid(0, end, knots, count=4096)
    if all(abs(interpolate(x)-finite_value(fn, x)) <= 2e-13*max(scale, 1e-300) for x in check):
        return {"kind": "table", "points": points}
    return {"kind": "source", "reason": "Waveform does not match a supported family at 2e-13 relative to peak"}


def profile_regions(blocks):
    mesh, geometry = blocks.get("Mesh", {}), blocks.get("Geometry", {})
    lo, hi = mesh.get("r_min", 0.0), mesh.get("r_max", 0.05)
    if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)) or hi <= lo:
        return None
    is2d = blocks.get("Main", {}).get("dimension") == "2D_RZ"
    fractions = geometry.get("volfrac", {})
    if not isinstance(fractions, dict) or not fractions:
        return None
    functions = [geometry.get(k) for k in ("rho", "Te", "Ti")] + list(fractions.values())
    if any(f is None for f in functions):
        return None
    def at(x):
        return tuple(finite_value(f, x, 0.0) if callable(f) and is2d else finite_value(f, x) for f in functions)
    hints = sorted({x for f in functions if callable(f) for x in numeric_hints(f) if lo < x < hi})
    xs = grid(lo, hi, hints)
    values = [at(x) for x in xs]
    # Continuous profiles stay in source. This bound also limits hostile inputs.
    if len(set(values)) > 256:
        return None
    bounds = [lo]
    for i in range(1, len(xs)):
        if values[i] == values[i-1]:
            continue
        a, b = xs[i-1], xs[i]
        for _ in range(80):
            mid = (a+b)/2
            if mid in (a, b):
                break
            if at(mid) == values[i-1]:
                a = mid
            else:
                b = mid
        if b > bounds[-1] and b < hi:
            bounds.append(b)
    bounds.append(hi)
    regions = []
    names = list(fractions)
    for a, b in zip(bounds, bounds[1:]):
        value = at((a+b)/2)
        if any(not close(at(a+(b-a)*fraction), value) for fraction in (0.125,0.375,0.625,0.875)):
            return None
        vf = value[3:]
        if sum(v == 1 for v in vf) != 1 or any(v not in (0, 1) for v in vf):
            return None
        if min(value[:3]) <= 0:
            return None
        if is2d:
            for z in (mesh.get("z_min", -hi), mesh.get("z_max", hi)):
                if any(not close(finite_value(f, (a+b)/2, z), expected) for f, expected in zip(functions, value) if callable(f)):
                    return None
        regions.append({"rOuter": b, "rho": value[0], "Te": value[1], "Ti": value[2], "materialName": names[vf.index(1)]})
    return regions


def spherical_shapes(blocks):
    if blocks.get("Main", {}).get("dimension") != "2D_RZ":
        return None
    geometry = blocks.get("Geometry", {})
    fractions = geometry.get("volfrac", {})
    if not isinstance(fractions, dict):
        return None
    functions = [geometry.get(k) for k in ("rho", "Te", "Ti")] + list(fractions.values())
    if any(f is None for f in functions):
        return None
    wrapped = {k: (lambda r, fn=geometry[k]: finite_value(fn, r, 0)) for k in ("rho", "Te", "Ti")}
    wrapped["volfrac"] = {name: (lambda r, fn=fn: finite_value(fn, r, 0)) for name, fn in fractions.items()}
    # Wrapper globals do not contain the source's numeric hints; retain the
    # constants as an explicit grid in the mesh for verification below.
    radial = dict(blocks, Main=dict(blocks["Main"], dimension="1D_SPH"), Geometry=wrapped)
    regions = profile_regions(radial)
    if not regions or len(regions) < 2:
        return None
    hi = blocks.get("Mesh", {}).get("r_max", 0.05)
    rs = grid(0, hi, [r["rOuter"] for r in regions], count=256)
    mesh = blocks.get("Mesh", {})
    for s in rs:
        for i in range(17):
            theta = math.pi*i/16
            r, z = s*math.sin(theta), s*math.cos(theta)
            if not mesh.get("z_min", -hi) <= z <= mesh.get("z_max", hi):
                continue
            # Boundary rounding at hypot() is handled by the final comparison.
            if any(abs(s-bound["rOuter"]) <= hi*1e-14 for bound in regions):
                continue
            if any(not close(finite_value(fn, r, z), finite_value(fn, s, 0)) for fn in functions):
                return None
    shapes, inner = [], 0.0
    for region in regions[:-1]:
        outer = math.nextafter(region["rOuter"], -math.inf)
        params = {"z0":0.0, "radius":outer}
        kind = "solidSphere" if inner == 0 else "shell"
        if inner:
            params["rIn"] = inner
        shapes.append([kind,region["materialName"],region["rho"],region["Te"],region["Ti"],params])
        inner = region["rOuter"]
    bg = regions[-1]
    return shapes, [bg["materialName"],bg["rho"],bg["Te"],bg["Ti"]]


def encode(value, path=(), end=1e-9):
    if callable(value):
        result = {"_type": "callable"}
        if path and path[0] != "Geometry":
            try:
                result["curve"] = curve(value, end)
            except Exception as error:
                result["curve"] = {"kind": "source", "reason": str(error)}
        return result
    if value is None or type(value) in (str, bool):
        return value
    if hasattr(value, "tolist"):
        return encode(value.tolist(), path, end)
    if type(value) in (float, int):
        if not math.isfinite(value):
            return {"_type": "unsupported", "reason": "Non-finite number"}
        if isinstance(value, int) and abs(value) > 2**53-1:
            return {"_type": "integer", "value": str(value)}
        return value
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: encode(v, path+(k,), end) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v, path+(i,), end) for i, v in enumerate(value)]
    return {"_type": "unsupported", "reason": "Python value of type " + type(value).__name__}


def evaluate(source, filename):
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    return _studio_capture(source, filename)


def sample_arguments(blocks, path, functions):
    end = blocks.get("Main", {}).get("t_end", 1e-9)
    hints = {x for fn in functions if callable(fn) for x in numeric_hints(fn)}
    if path[0] != "Geometry":
        return [(x,) for x in grid(0, end, hints, 4096)]
    mesh = blocks.get("Mesh", {})
    lo, hi = mesh.get("r_min", 0), mesh.get("r_max", mesh.get("box_r_max", 0.05))
    nodes = mesh.get("explicit_nodes", [])
    if isinstance(nodes, (list, tuple)):
        hints.update(x for x in nodes if isinstance(x, (int, float)))
    # Mesh nodes and centres, plus dense samples and located profile boundaries.
    nr = max(1, int(mesh.get("nr", 300)))
    hints.update(lo+(hi-lo)*i/nr for i in range(nr+1))
    hints.update(lo+(hi-lo)*(i+0.5)/nr for i in range(nr))
    recovered = profile_regions(blocks)
    if recovered:
        hints.update(r["rOuter"] for r in recovered)
    rs = grid(lo, hi, hints)
    try:
        physical = mesh_points(blocks)
    except (ArithmeticError, TypeError, ValueError, KeyError):
        physical = []  # Dense cuts still cover decks with solver-only mesh rules.
    if blocks.get("Main", {}).get("dimension") != "2D_RZ":
        return [(r,) for r in rs] + physical
    zlo, zhi = mesh.get("z_min", mesh.get("box_z_min", -hi)), mesh.get("z_max", mesh.get("box_z_max", hi))
    nz = max(1, int(mesh.get("nz", 64)))
    zs = sorted(set([zlo, zhi, 0.0] + [zlo+(zhi-zlo)*(i+0.5)/nz for i in range(nz)]))
    # Every logical cell centre at the declared resolution, plus grid nodes and
    # dense cuts. Curvilinear decks also sample the explicit radial nodes above.
    result = [(lo+(hi-lo)*(i+0.5)/nr, z) for i in range(nr) for z in zs]
    result.extend((lo+(hi-lo)*i/nr, zlo+(zhi-zlo)*j/nz) for i in range(nr+1) for j in range(nz+1))
    result.extend((r, z) for r in rs for z in (zlo, (3*zlo+zhi)/4, 0, (zlo+3*zhi)/4, zhi))
    result.extend(physical)
    return result


def compare_values(original, candidate, blocks, path, arguments=None):
    if callable(original) or callable(candidate):
        if not callable(original) or not callable(candidate):
            return False, "Literal/callable distinction retained"
        try:
            points = arguments if arguments is not None else sample_arguments(blocks, path, (original, candidate))
            for args in points:
                if not close(finite_value(original, *args), finite_value(candidate, *args)):
                    return False, "Function differs at " + repr(args)
            return True, "Function verified on mesh/time samples and knot neighbours (relative tolerance 1e-12); finite sampling is not a proof"
        except Exception as error:
            return False, "Sampling failed: " + str(error)
    if close(original, candidate):
        return True, "Literal value"
    def short_value(value):
        formatter = reprlib.Repr()
        formatter.maxstring = formatter.maxother = 120
        text = formatter.repr(value)
        return text if len(text) <= 160 else text[:157] + "..."
    return False, "Source value retained: the generated deck would set {0} instead of {1}".format(short_value(candidate), short_value(original))


def make_rules(original, candidate, context=None):
    rules = []
    missing = object()
    geometry_arguments = None
    def callables(value):
        if callable(value):
            return [value]
        if isinstance(value, dict):
            return [f for v in value.values() for f in callables(v)]
        if isinstance(value, (tuple,list)):
            return [f for v in value for f in callables(v)]
        return []
    def walk(old, new, path):
        nonlocal geometry_arguments
        if old is missing:
            rules.append({"path": list(path), "kind": "omitted", "reason": "Absent from source; GUI default is not added"})
        elif new is missing:
            # The form did not write this key (for example because the GUI
            # omits solver defaults). sourceOnly lets an edit of the form field
            # bound to exactly this path replace the retained source value.
            rules.append({"path": list(path), "kind": "passthrough", "reason": "Outside the form's emitted settings", "sourceOnly": True})
        elif isinstance(old, dict) and isinstance(new, dict):
            if not old and not new:
                rules.append({"path": list(path), "kind": "mapped", "reason": "Empty block/dictionary"})
            for key in dict.fromkeys(list(old)+list(new)):
                walk(old.get(key, missing), new.get(key, missing), path+(key,))
        elif isinstance(old, (list, tuple)) and type(old) is type(new) and len(old) == len(new) and any(isinstance(x, dict) for x in old):
            for i, (a, b) in enumerate(zip(old, new)):
                walk(a, b, path+(i,))
        else:
            if path[0] == "Geometry" and callable(old) and callable(new) and geometry_arguments is None:
                geometry_arguments = sample_arguments(context or original,path,callables(original.get("Geometry",{}))+callables(candidate.get("Geometry",{})))
            same, reason = compare_values(old, new, context or original, path, geometry_arguments if path[0]=="Geometry" else None)
            rules.append({"path": list(path), "kind": ("approximated" if callable(old) else "mapped") if same else "passthrough", "reason": reason})
    walk(original, candidate, ())
    return rules


def worker(request):
    source, filename = request["source"], request["filename"]
    blocks, namespace, calls = evaluate(source, filename)
    if "Main" not in blocks:
        raise ValueError("No Main block was executed; this file is not a namelist deck")
    operation = request.get("operation", "record")
    repeated = []
    for name, history in calls.items():
        # Mesh resets arrays/topology even when a later call omits those keys.
        # Preserve all non-Main repeated setters instead of assuming uniform
        # merge semantics across Builder implementations. Main is scalar-only.
        if name != "Main" and len(history) > 1:
            repeated.append(name)
            continue
        seen = set()
        for arguments in history:
            if any(key in seen and isinstance(value, (dict,list,tuple)) for key,value in arguments.items()):
                repeated.append(name)
                break
            seen.update(arguments)
    if operation in ("verify", "compare"):
        candidate, _, candidate_calls = evaluate(request["candidate"], str(Path(filename).with_name("__studio_candidate__.py")))
        rules = make_rules(blocks, candidate)
        if operation == "compare":
            failures = [r for r in rules if r["kind"] not in ("mapped", "approximated")]
            for name in repeated:
                history = candidate_calls.get(name, [])
                if len(history) != len(calls[name]):
                    failures.append({"path":[name],"kind":"passthrough","reason":"Repeated block calls were flattened"})
                else:
                    for old,new in zip(calls[name],history):
                        failures.extend(r for r in make_rules({name:old},{name:new},blocks) if r["kind"] not in ("mapped","approximated"))
            return {"ok": not failures, "failures": failures, "rules": rules}
        rules = [r for r in rules if r["path"][0] not in repeated]
        rules.extend({"path":[name],"kind":"passthrough","reason":"Repeated block calls retain their original order, including setter resets; solver overwrite warnings may remain"} for name in repeated)
        return {"ok": True, "rules": rules}
    end = blocks["Main"].get("t_end", 1e-9)
    result = {"ok": True, "blocks": encode(blocks, end=end), "regions": None, "meshSampling": sampling_note(blocks)}
    try:
        corona = corona_profile(blocks)
        if corona and "VOID" in blocks.get("Geometry", {}).get("volfrac", {}):
            start = corona["start"]
            geometry = dict(blocks["Geometry"])
            geometry["rho"] = lambda r: finite_value(blocks["Geometry"]["rho"], r) if r < start else 1e-10
            geometry["volfrac"] = {
                name: (lambda r, name=name, fn=fn: finite_value(fn, r) if r < start else float(name == "VOID"))
                for name, fn in blocks["Geometry"]["volfrac"].items()
            }
            result["regions"] = profile_regions(dict(blocks, Geometry=geometry))
            result["corona"] = corona
        else:
            result["regions"] = profile_regions(blocks)
    except Exception as error:
        result["profileReason"] = str(error)
    # These are executed geometry parameters, not the GUI state header. Arbitrary
    # shape functions remain source-preserved when no explicit shape list exists.
    if "_GUI_SHAPES" in namespace and "_GUI_BG" in namespace:
        result["shapes"] = encode(namespace["_GUI_SHAPES"])
        result["background"] = encode(namespace["_GUI_BG"])
    elif result["regions"] is None:
        try:
            shapes = spherical_shapes(blocks)
            if shapes:
                result["shapes"], result["background"] = shapes
        except Exception:
            pass  # The full source is retained if geometric inference fails.
    plan = namespace.get("PLAN")
    if blocks.get("Mesh", {}).get("logical_mesh_2d") == "polar_in_box" and hasattr(plan, "mesh_kwargs"):
        base_nodes = plan.mesh_kwargs.get("explicit_nodes", [])
        nodes = blocks["Mesh"].get("explicit_nodes", [])
        if len(base_nodes) >= 2 and len(nodes) >= len(base_nodes):
            tail = len(nodes)-len(base_nodes)
            ratio = (nodes[len(base_nodes)]-nodes[len(base_nodes)-1])/(base_nodes[-1]-base_nodes[-2]) if tail else 1.15
            count = len(base_nodes)-1
            for warning in getattr(plan, "report", {}).get("warnings", []):
                requested = re.search(r"smooth zoning n_total is advisory: requested (\d+)", warning)
                if requested:
                    count = int(requested.group(1))
            result["planner"] = {"radialCount":count,"tailRings":tail,"tailRatio":ratio}
    return result


def run_request(request, timeout=30):
    """Run the recorder with a hard timeout and a separate JSON response file."""
    request = dict(request)
    working_directory = str(Path(request.get("workingDirectory") or Path(request["filename"]).resolve().parent).expanduser().resolve())
    if not Path(working_directory).is_dir():
        return {"ok": False, "error": "Import working directory does not exist: " + working_directory}
    filename = Path(request["filename"]).expanduser()
    request["filename"] = str(filename if filename.is_absolute() else Path(working_directory)/filename)
    overrides = request.get("environment", {})
    if not isinstance(overrides, dict) or any(not isinstance(k,str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*",k) or not isinstance(v,str) or "\0" in v for k,v in overrides.items()):
        return {"ok": False, "error": "Import environment must contain KEY=VALUE strings without NUL bytes"}
    environment = dict(os.environ, **overrides)
    if request.get("repoRoot"):
        environment["TENRYU_REPO"] = overrides.get("TENRYU_REPO", request["repoRoot"])
        environment["PYTHONPATH"] = request["repoRoot"] + os.pathsep + environment.get("PYTHONPATH", "")
    with tempfile.TemporaryDirectory(prefix="tenryu-import-") as directory:
        request_path = Path(directory)/"request.json"
        result_path = Path(directory)/"result.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(request_path), str(result_path)],
                                   cwd=working_directory, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            return {"ok": False, "error": "Deck evaluation exceeded {0} seconds".format(timeout)}
        if not result_path.exists():
            return {"ok": False, "error": "Python recorder exited {0}: {1}".format(process.returncode, stderr.decode(errors="replace")[-8000:])}
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["stdout"] = stdout.decode(errors="replace")[-8000:]
        result["stderr"] = stderr.decode(errors="replace")[-8000:]
        result["workingDirectory"] = working_directory
        result["filename"] = request["filename"]
        return result


def main_import_deck(args):
    try:
        if args.request:
            request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        else:
            path = Path(args.deck).resolve()
            request = {"source": path.read_text(encoding="utf-8"), "filename": str(path)}
        if args.repo_root:
            root = Path(args.repo_root).resolve()
            if not (root/"tools"/"mesh_planner.py").is_file():
                raise ValueError("TENRYU mirror is missing tools/mesh_planner.py: " + str(root))
            request["repoRoot"] = str(root)
        result = run_request(request, args.timeout)
    except Exception:
        result = {"ok": False, "error": traceback.format_exc()}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    # Direct worker invocation also works from a bundled tools/assist directory.
    request = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    try:
        result = worker(request)
    except BaseException:
        result = {"ok": False, "error": traceback.format_exc()}
    Path(sys.argv[3]).write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
