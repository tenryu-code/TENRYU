"""Deck linting through TENRYU's validate and freeze commands."""

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set


_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_NONFINITE_RE = re.compile(
    r'(?<=[:,\[\s])(-?)(?:nan|inf(?:inity)?)(?=[,}\]\s]|$)'
)
_SKIPPED_DEFAULT_KEYS = {
    "_frozen_at",
    "_tenryu_version",
    "frozen_summary",
    "_namelist_source_hash",
}
_REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_mesh_preview(stdout_text: str) -> Optional[dict]:
    """Parse the last mesh-preview JSON record from solver output."""
    prefix = "TENRYU-MESH-PREVIEW: "
    payload = None
    for line in stdout_text.splitlines():
        if line.startswith(prefix):
            payload = line[len(prefix) :]
    if payload is None:
        return None
    payload = _NONFINITE_RE.sub(
        lambda m: m.group(1) + "Infinity" if "inf" in m.group(0) else "NaN",
        payload,
    )
    return json.loads(payload)


def parse_autozone_stats(log_text: str) -> Optional[dict]:
    """Parse the last mesh-autozone statistics line."""
    selected = None
    for line in log_text.splitlines():
        if "[mesh-autozone]" in line:
            selected = line
    if selected is None:
        return None

    mass_values = []
    mass_position = selected.find("mass_ratio")
    if mass_position >= 0:
        mass_values = [
            float(value)
            for value in re.findall(
                _FLOAT_PATTERN, selected[mass_position + len("mass_ratio") :]
            )
        ]

    dr_value = None
    dr_match = re.search(
        r"dr_min_actual[^0-9+\-.]*({0})".format(_FLOAT_PATTERN), selected
    )
    if dr_match is not None:
        dr_value = float(dr_match.group(1))

    return {
        "mass_ratio_min": mass_values[0] if len(mass_values) > 0 else None,
        "mass_ratio_max": mass_values[1] if len(mass_values) > 1 else None,
        "mass_ratio_mean": mass_values[2] if len(mass_values) > 2 else None,
        "dr_min_actual": dr_value,
    }


def pin_exempt_edges(r_nodes, zoning_pins) -> Set[int]:
    """Return adjacent-width edge indices exempted by allowed zoning pins."""
    exempt_edges = set()
    if not isinstance(zoning_pins, list) or len(r_nodes) < 2:
        return exempt_edges
    tolerance = 1.0e-12 * max(1.0, abs(r_nodes[-1] - r_nodes[0]))
    for pin in zoning_pins:
        if not isinstance(pin, dict) or not pin.get("ratio_jump_allowed"):
            continue
        for node_index in range(1, len(r_nodes) - 1):
            if abs(r_nodes[node_index] - pin["r"]) <= tolerance:
                exempt_edges.add(node_index - 1)
                break
    return exempt_edges


def geom_ratio_lint(
    r_nodes: List[float], exempt_edges: Optional[Set[int]] = None
) -> dict:
    """Check node monotonicity and adjacent cell-width ratios."""
    if exempt_edges is None:
        exempt_edges = set()
    n_cells = max(0, len(r_nodes) - 1)
    widths = []
    for index in range(n_cells):
        width = r_nodes[index + 1] - r_nodes[index]
        if width <= 0.0:
            return {
                "monotonic_ok": False,
                "n_cells": n_cells,
                "max_adjacent_dr_ratio": None,
                "at_index": None,
                "pin_jump_edges": [],
                "n_exempt_edges": 0,
            }
        widths.append(width)

    maximum = None
    maximum_index = None
    pin_jump_edges = []
    for index in range(len(widths) - 1):
        ratio = max(
            widths[index + 1] / widths[index],
            widths[index] / widths[index + 1],
        )
        if index in exempt_edges:
            pin_jump_edges.append({"at_index": index, "ratio": ratio})
            continue
        if maximum is None or ratio > maximum:
            maximum = ratio
            maximum_index = index
    return {
        "monotonic_ok": True,
        "n_cells": n_cells,
        "max_adjacent_dr_ratio": maximum,
        "at_index": maximum_index,
        "pin_jump_edges": pin_jump_edges,
        "n_exempt_edges": len(pin_jump_edges),
    }


def width_ratio_lint_entry(geometry: dict, zoning_measure) -> dict:
    """Build the adjacent-width-ratio lint entry."""
    mass_measure = zoning_measure is not None and zoning_measure != "width"
    ratio = geometry["max_adjacent_dr_ratio"]
    entry = {
        "id": "adjacent-dr-ratio",
        "severity": "info" if mass_measure else "warn",
        "ok": (
            True
            if mass_measure
            else (
                (ratio <= 1.5)
                if ratio is not None
                else (geometry["n_exempt_edges"] > 0)
            )
        ),
        "detail": geometry,
    }
    if mass_measure:
        entry["note"] = (
            "width ratios induced by measure '{0}'; cell-measure ratio contract "
            "verified by the zoning solver"
        ).format(zoning_measure)
    return entry


def _resolve_pin_path(values: dict, path: str):
    current = values
    for component in path.split("."):
        match = re.fullmatch(r"([^\[\]]+)\[(\d+)\]", component)
        if match is None:
            if not isinstance(current, dict) or component not in current:
                return False, None
            current = current[component]
            continue
        name = match.group(1)
        index = int(match.group(2))
        if not isinstance(current, dict) or name not in current:
            return False, None
        current = current[name]
        if not isinstance(current, list) or index >= len(current):
            return False, None
        current = current[index]
    return True, current


def intent_lock_check(frozen: dict, pins: Dict[str, object]) -> List[dict]:
    """Compare selected frozen-config paths against expected values."""
    results = []
    for path, expected in pins.items():
        found, actual = _resolve_pin_path(frozen, path)
        if not found:
            results.append(
                {
                    "path": path,
                    "expected": expected,
                    "actual": None,
                    "ok": False,
                    "reason": "path not found",
                }
            )
            continue
        if isinstance(actual, float) or isinstance(expected, float):
            try:
                ok = math.isclose(
                    float(actual), float(expected), rel_tol=1.0e-9, abs_tol=1.0e-300
                )
            except (TypeError, ValueError, OverflowError):
                ok = False
        else:
            ok = actual == expected
        results.append(
            {"path": path, "expected": expected, "actual": actual, "ok": ok}
        )
    return results


def _leaf_values(values: dict) -> dict:
    leaves = {}

    def walk(current, components):
        if isinstance(current, dict):
            for key, value in current.items():
                if not components and key in _SKIPPED_DEFAULT_KEYS:
                    continue
                walk(value, components + [key])
            return
        leaves[".".join(components)] = current

    walk(values, [])
    return leaves


def defaults_diff(frozen: dict, baseline: dict) -> dict:
    """Compare frozen configuration leaves with a baseline."""
    target_leaves = _leaf_values(frozen)
    baseline_leaves = _leaf_values(baseline)
    target_paths = set(target_leaves)
    baseline_paths = set(baseline_leaves)
    common_paths = sorted(target_paths & baseline_paths)
    all_differs = []
    equal_count = 0
    for path in common_paths:
        if target_leaves[path] == baseline_leaves[path]:
            equal_count += 1
        else:
            all_differs.append(
                {
                    "path": path,
                    "value": target_leaves[path],
                    "baseline": baseline_leaves[path],
                }
            )

    result = {
        "differs": all_differs[:400],
        "equal_leaf_count": equal_count,
        "only_in_target": sorted(target_paths - baseline_paths),
        "only_in_baseline": sorted(baseline_paths - target_paths),
        "heuristic_note": "leaves equal to the baseline are only PROBABLY defaults",
    }
    if len(all_differs) > 400:
        result["truncated"] = True
    return result


def _resolve_tenryu(requested: Optional[str]) -> Optional[str]:
    if requested is None:
        path = _REPO_ROOT / "build" / "tenryu"
    else:
        path = Path(requested).expanduser().resolve()
    if not path.is_file():
        return None
    return str(path)


def _missing_binary_error(requested: Optional[str]) -> None:
    if requested is None:
        shown = str(_REPO_ROOT / "build" / "tenryu")
    else:
        shown = str(Path(requested).expanduser().resolve())
    print(
        "assist: tenryu binary not found: {0}; build ./build/tenryu or use "
        "--tenryu PATH".format(shown),
        file=sys.stderr,
    )


def _write_output(payload: dict, output: Optional[str]) -> bool:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(rendered, end="")
        return True
    try:
        with open(output, "w", encoding="utf-8") as stream:
            stream.write(rendered)
    except OSError as error:
        print("assist: cannot write output: {0}".format(error), file=sys.stderr)
        return False
    return True


def _load_json(path: str, label: str):
    try:
        with open(path, "r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError) as error:
        raise ValueError("{0}: {1}".format(label, error)) from error


def _load_lint_inputs(intent_path=None, baseline_path=None):
    pins = None
    if intent_path is not None:
        intent = _load_json(intent_path, "cannot load intent JSON")
        if (
            not isinstance(intent, dict)
            or set(intent) != {"pins"}
            or not isinstance(intent["pins"], dict)
        ):
            raise ValueError(
                'intent JSON must have schema {"pins": {...}}'
            )
        pins = intent["pins"]

    baseline = None
    if baseline_path is not None:
        baseline = _load_json(baseline_path, "cannot load baseline JSON")
        if not isinstance(baseline, dict):
            raise ValueError("baseline JSON must be an object")

    return pins, baseline


def run_deck_lint(deck: str, tenryu: str, pins=None, baseline=None, keep_tmp=False):
    """Run validate/freeze lints; returns (payload_dict, exit_code_int)."""
    deck = os.path.abspath(deck)

    temporary = tempfile.mkdtemp(prefix="assist_lint_")
    try:
        try:
            validate = subprocess.run(
                [tenryu, "validate", deck, "--mesh-preview"],
                cwd=str(_REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=600,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return (
                {
                    "schema": "tenryu.assist.decklint.v0",
                    "deck": deck,
                    "tenryu": tenryu,
                    "error": "validate failed to run: {0}".format(error),
                },
                2,
            )

        combined = validate.stdout + validate.stderr
        try:
            preview = parse_mesh_preview(validate.stdout)
            if preview is None:
                preview = parse_mesh_preview(combined)
        except ValueError as error:
            return (
                {
                    "schema": "tenryu.assist.decklint.v0",
                    "deck": deck,
                    "tenryu": tenryu,
                    "error": "malformed mesh preview JSON: {0}".format(error),
                },
                2,
            )
        autozone = parse_autozone_stats(combined)

        frozen = None
        if validate.returncode == 0:
            frozen_path = os.path.join(temporary, "frozen.json")
            try:
                freeze = subprocess.run(
                    [tenryu, "freeze", deck, "-o", frozen_path],
                    cwd=str(_REPO_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
            except (OSError, subprocess.TimeoutExpired):
                freeze = None
            if freeze is not None and freeze.returncode == 0 and os.path.isfile(frozen_path):
                try:
                    loaded = _load_json(frozen_path, "cannot load frozen JSON")
                except ValueError:
                    loaded = None
                if isinstance(loaded, dict):
                    frozen = loaded

        lints = []
        preview_summary = None
        if isinstance(preview, dict):
            r_nodes = preview.get("r_nodes")
            zoning_measure = preview.get("zoning_measure")
            preview_summary = {
                "nr": preview.get("nr"),
                "r_min": preview.get("r_min"),
                "r_max": preview.get("r_max"),
                "n_r_nodes": len(r_nodes) if isinstance(r_nodes, list) else None,
            }
            if isinstance(r_nodes, list):
                exempt_edges = pin_exempt_edges(
                    r_nodes, preview.get("zoning_pins")
                )
                geometry = geom_ratio_lint(r_nodes, exempt_edges)
                lints.append(
                    {
                        "id": "node-monotonic",
                        "severity": "hard",
                        "ok": geometry["monotonic_ok"],
                    }
                )
                lints.append(
                    width_ratio_lint_entry(geometry, zoning_measure)
                )
                if preview.get("nr") is not None:
                    lints.append(
                        {
                            "id": "nr-consistency",
                            "severity": "warn",
                            "ok": preview["nr"] == len(r_nodes) - 1,
                        }
                    )

        if autozone is not None:
            mass_ratio_max = autozone["mass_ratio_max"]
            lints.append(
                {
                    "id": "autozone-mass-ratio",
                    "severity": (
                        "hard"
                        if mass_ratio_max is not None and mass_ratio_max > 2.0
                        else "warn"
                    ),
                    "ok": mass_ratio_max is not None and mass_ratio_max <= 1.3,
                    "detail": autozone,
                }
            )

        intent_lock = None
        if pins is not None:
            if frozen is None:
                intent_lock = [
                    {
                        "path": path,
                        "expected": expected,
                        "actual": None,
                        "ok": False,
                        "reason": "freeze unavailable",
                    }
                    for path, expected in pins.items()
                ]
            else:
                intent_lock = intent_lock_check(frozen, pins)

        default_changes = None
        if baseline is not None and frozen is not None:
            default_changes = defaults_diff(frozen, baseline)

        payload = {
            "schema": "tenryu.assist.decklint.v0",
            "deck": deck,
            "tenryu": tenryu,
            "validate": {
                "exit_code": validate.returncode,
                "ok": validate.returncode == 0,
                "stderr_tail": validate.stderr[-2000:],
            },
            "mesh_preview": preview_summary,
            "lints": lints,
            "intent_lock": intent_lock,
            "defaults_diff": default_changes,
        }
        if validate.returncode != 0:
            return payload, 2
        if any(
            lint["severity"] == "hard" and not lint["ok"] for lint in lints
        ):
            return payload, 2
        if intent_lock is not None and any(not entry["ok"] for entry in intent_lock):
            return payload, 2
        return payload, 0
    finally:
        if not keep_tmp:
            shutil.rmtree(temporary)


def main_lint_deck(args) -> int:
    """CLI entry point for lint-deck."""
    tenryu = _resolve_tenryu(args.tenryu)
    if tenryu is None:
        _missing_binary_error(args.tenryu)
        return 2

    try:
        pins, baseline = _load_lint_inputs(args.intent, args.baseline)
    except ValueError as error:
        print("assist lint-deck: {0}".format(error), file=sys.stderr)
        return 2

    payload, exit_code = run_deck_lint(
        args.deck,
        tenryu,
        pins=pins,
        baseline=baseline,
        keep_tmp=args.keep_tmp,
    )
    if "error" in payload:
        print("assist lint-deck: {0}".format(payload["error"]), file=sys.stderr)
    if not _write_output(payload, args.output):
        return 2
    return exit_code


def main_freeze_baseline(args) -> int:
    """CLI entry point for freeze-baseline."""
    tenryu = _resolve_tenryu(args.tenryu)
    if tenryu is None:
        _missing_binary_error(args.tenryu)
        return 2
    deck = os.path.abspath(args.deck)
    output = os.path.abspath(args.output or "baseline_frozen.json")
    try:
        completed = subprocess.run(
            [tenryu, "freeze", deck, "-o", output],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(
            "assist freeze-baseline: freeze failed to run: {0}".format(error),
            file=sys.stderr,
        )
        return 2
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr[-2000:], file=sys.stderr, end="")
        return 2
    print(output)
    return 0
