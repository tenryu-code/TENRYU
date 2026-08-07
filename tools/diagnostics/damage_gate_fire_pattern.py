#!/usr/bin/env python3
"""Summarize Phase 9 ALE remap-damage gate fire patterns from TENRYU logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_RUN_GLOBS = ("build/output_h3b_phase9_11_*", "build/output_h3b_256_T31")
LAMBDA_SCHEDULE = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)

DAMAGE_RE = re.compile(
    r"\[ale-stats\] (?P<tag>damage|damage_accepted) D_rho_max=(?P<d>[-+0-9.eE]+) "
    r"A_axis_max=(?P<a>[-+0-9.eE]+) axis_max_j=(?P<j>-?\d+)"
)
BACKTRACK_RE = re.compile(
    r"\[ale-stats\] backtrack_lambda_accepted=(?P<lam>[-+0-9.eE]+) "
    r"trials=(?P<trials>\d+) first_fail_cell=\(c=(?P<c>-?\d+),i=(?P<i>-?\d+),j=(?P<j>-?\d+)\)"
)
PREVENTIVE_RE = re.compile(r"preventive axis guard triggered ALE at step (?P<step>\d+)")
OUTPUT_RE = re.compile(r"wrote (?:initial )?snapshot \(step=(?P<step>\d+), t=(?P<t>[-+0-9.eE]+)\)")
PHASE10_RE = re.compile(r"Phase 10 axis-spine-only repair: max\|dz\|=(?P<dz>[-+0-9.eE]+)")
AXIS_BYPASS_RE = re.compile(
    r"Phase 9 axis-emergency bypass: damage gate disabled this ALE invoke "
    r"\(axis_margin_min=(?P<margin>[-+0-9.eE]+)\)"
)
EMERGENCY_BYPASS_RE = re.compile(
    r"Phase 9 emergency bypass: damage gate rejected all candidates; "
    r"falling back to legacy criterion at lambda=(?P<lam>[-+0-9.eE]+)"
)


def finite_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value


def parse_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return math.nan


def find_log(path: Path) -> tuple[Path, Path]:
    if path.is_file():
        run_dir = path.parent.parent if path.parent.name == "log" else path.parent
        return run_dir, path
    preferred = path / "log" / "tenryu.log"
    if preferred.exists():
        return path, preferred
    logs = sorted(path.glob("log/*.log")) + sorted(path.glob("*.log"))
    if logs:
        return path, logs[0]
    raise FileNotFoundError(f"no log file found under {path}")


def default_inputs() -> list[Path]:
    out: list[Path] = []
    for pattern in DEFAULT_RUN_GLOBS:
        out.extend(sorted(Path(".").glob(pattern)))
    return out


def load_config(run_dir: Path) -> dict[str, Any]:
    config_files = sorted((run_dir / "config").glob("*_frozen.json"))
    if not config_files:
        return {}
    try:
        with config_files[0].open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {"config_path": str(config_files[0])}

    ale = data.get("numerics", {}).get("ale", {})
    mesh = data.get("mesh", {})
    main = data.get("main", {})
    return {
        "config_path": str(config_files[0]),
        "case_name": main.get("name"),
        "seed": main.get("seed"),
        "nr": mesh.get("nr"),
        "nz": mesh.get("nz"),
        "ale_every_n_steps": ale.get("every_n_steps"),
        "remap_damage_gate_enabled": ale.get("remap_damage_gate_enabled"),
        "remap_damage_dmax": ale.get("remap_damage_dmax"),
        "remap_damage_axis_eta": ale.get("remap_damage_axis_eta"),
        "remap_damage_axis_budget_enabled": ale.get("remap_damage_axis_budget_enabled"),
        "remap_damage_axis_budget_factor": ale.get("remap_damage_axis_budget_factor"),
        "axis_repair_mode": ale.get("axis_repair_mode"),
        "remap_scheme": ale.get("remap_scheme"),
        "remap_ms2_limiter": ale.get("remap_ms2_limiter"),
    }


def new_event(index: int, line_no: int, lower_step: int | None, lower_t: float | None) -> dict[str, Any]:
    return {
        "event_index": index,
        "start_line": line_no,
        "end_line": line_no,
        "step": None,
        "step_source": "not_emitted",
        "step_lower_bound": lower_step,
        "t_lower_bound_s": finite_or_none(lower_t),
        "trigger": "regular_or_unknown",
        "repair_max_dz": None,
        "damages": [],
        "accepted_lambda": None,
        "accepted_trials": None,
        "first_fail_cell": None,
        "axis_emergency_bypass": False,
        "axis_emergency_margin": None,
        "phase9_emergency_bypass": False,
        "phase9_emergency_lambda": None,
        "incomplete": True,
    }


def assign_pending_trigger(event: dict[str, Any], pending_trigger: dict[str, Any] | None) -> None:
    if pending_trigger is None:
        return
    event["step"] = pending_trigger["step"]
    event["step_source"] = "preventive_axis_guard_log"
    event["trigger"] = "preventive_axis_guard"
    event["trigger_line"] = pending_trigger["line_no"]


def classify_damage(dmg: dict[str, Any], dmax: float | None, axis_eta: float | None) -> dict[str, Any]:
    gate_outcome = dmg.get("gate_outcome", "rejected")
    d_excess = dmg["D_rho_max"] - dmax if dmax is not None else None
    a_excess = dmg["A_axis_max"] - axis_eta if axis_eta is not None else None
    d_fired = d_excess is not None and d_excess > 0.0
    axis_fired = a_excess is not None and a_excess > 0.0

    reasons: list[str] = []
    if gate_outcome == "accepted":
        if d_fired or axis_fired:
            reasons.append("missed_fire_candidate")
        else:
            reasons.append("accepted_below_threshold")
    else:
        if d_fired:
            reasons.append("D_rho_max")
        if axis_fired:
            reasons.append("A_axis_max")
        if not reasons:
            reasons.append("axis_budget_or_unclassified")

    axis_j = dmg["axis_max_j"]
    if axis_fired and axis_j >= 0:
        location = {"kind": "axis_row", "i": 0, "j": axis_j}
    elif d_fired or gate_outcome == "rejected":
        location = {
            "kind": "global_D_rho_max_unlocated",
            "note": "current production log does not emit the cell for D_rho_max",
        }
    else:
        location = {"kind": "damage_gate_accepted"}

    return {
        **dmg,
        "D_rho_excess": finite_or_none(d_excess),
        "A_axis_excess": finite_or_none(a_excess),
        "reasons": reasons,
        "location": location,
        "false_fire_candidate": gate_outcome == "rejected" and not (d_fired or axis_fired),
        "missed_fire_candidate": gate_outcome == "accepted" and (d_fired or axis_fired),
    }


def infer_missing_steps(events: list[dict[str, Any]], triggers: list[dict[str, int]], every_n: int | None) -> None:
    trigger_by_line = sorted(triggers, key=lambda item: item["line_no"])
    for idx, event in enumerate(events):
        if event["step"] is not None:
            continue
        next_event_line = events[idx + 1]["start_line"] if idx + 1 < len(events) else None
        next_triggers = [
            item
            for item in trigger_by_line
            if item["line_no"] > event["end_line"]
            and (next_event_line is None or item["line_no"] < next_event_line)
        ]
        if next_triggers:
            event["step"] = next_triggers[0]["step"] - 1
            event["step_source"] = "next_preventive_axis_guard_minus_1"
            event["trigger"] = "regular_cadence_before_preventive_guard"
            continue

        if every_n is not None and every_n > 0:
            previous_steps = [e["step"] for e in events[:idx] if e.get("step") is not None]
            if previous_steps:
                prev = max(previous_steps)
                event["step"] = ((prev // every_n) + 1) * every_n
                event["step_source"] = "cadence_inferred_from_previous_event"
                event["trigger"] = "regular_cadence_inferred"


def parse_log(run_dir: Path, log_path: Path, config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    dmax = args.dmax if args.dmax is not None else config.get("remap_damage_dmax")
    axis_eta = args.axis_eta if args.axis_eta is not None else config.get("remap_damage_axis_eta")
    dmax = float(dmax) if dmax is not None else None
    axis_eta = float(axis_eta) if axis_eta is not None else None

    events: list[dict[str, Any]] = []
    triggers: list[dict[str, int]] = []
    current: dict[str, Any] | None = None
    pending_trigger: dict[str, int] | None = None
    pending_repair_dz: float | None = None
    last_output_step: int | None = None
    last_output_t: float | None = None

    def ensure_event(line_no: int) -> dict[str, Any]:
        nonlocal current, pending_trigger, pending_repair_dz
        if current is None:
            current = new_event(len(events) + 1, line_no, last_output_step, last_output_t)
            assign_pending_trigger(current, pending_trigger)
            pending_trigger = None
            current["repair_max_dz"] = finite_or_none(pending_repair_dz)
            pending_repair_dz = None
        current["end_line"] = line_no
        return current

    def finish_event(line_no: int) -> None:
        nonlocal current
        if current is None:
            return
        current["end_line"] = line_no
        current["incomplete"] = False
        current["damages"] = [
            classify_damage(dmg, dmax, axis_eta) for dmg in current["damages"]
        ]
        events.append(current)
        current = None

    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            if match := OUTPUT_RE.search(line):
                last_output_step = int(match.group("step"))
                last_output_t = parse_float(match.group("t"))

            if match := PREVENTIVE_RE.search(line):
                pending_trigger = {"step": int(match.group("step")), "line_no": line_no}
                triggers.append(pending_trigger)

            if match := PHASE10_RE.search(line):
                pending_repair_dz = parse_float(match.group("dz"))

            if match := AXIS_BYPASS_RE.search(line):
                event = ensure_event(line_no)
                event["axis_emergency_bypass"] = True
                event["axis_emergency_margin"] = parse_float(match.group("margin"))

            if match := DAMAGE_RE.search(line):
                event = ensure_event(line_no)
                trial_index = len(event["damages"]) + 1
                gate_outcome = "accepted" if match.group("tag") == "damage_accepted" else "rejected"
                event["damages"].append(
                    {
                        "line": line_no,
                        "trial_index": trial_index,
                        "trial_lambda": LAMBDA_SCHEDULE[trial_index - 1]
                        if trial_index <= len(LAMBDA_SCHEDULE)
                        else None,
                        "gate_outcome": gate_outcome,
                        "D_rho_max": parse_float(match.group("d")),
                        "A_axis_max": parse_float(match.group("a")),
                        "axis_max_j": int(match.group("j")),
                    }
                )

            if match := EMERGENCY_BYPASS_RE.search(line):
                event = ensure_event(line_no)
                event["phase9_emergency_bypass"] = True
                event["phase9_emergency_lambda"] = parse_float(match.group("lam"))

            if match := BACKTRACK_RE.search(line):
                event = ensure_event(line_no)
                event["accepted_lambda"] = parse_float(match.group("lam"))
                event["accepted_trials"] = int(match.group("trials"))
                event["first_fail_cell"] = {
                    "c": int(match.group("c")),
                    "i": int(match.group("i")),
                    "j": int(match.group("j")),
                }
                finish_event(line_no)

    if current is not None:
        current["damages"] = [classify_damage(dmg, dmax, axis_eta) for dmg in current["damages"]]
        events.append(current)

    every_n = config.get("ale_every_n_steps")
    every_n = int(every_n) if isinstance(every_n, int) or str(every_n).isdigit() else None
    infer_missing_steps(events, triggers, every_n)

    return summarize_run(run_dir, log_path, config, events, dmax, axis_eta)


def summarize_run(
    run_dir: Path,
    log_path: Path,
    config: dict[str, Any],
    events: list[dict[str, Any]],
    dmax: float | None,
    axis_eta: float | None,
) -> dict[str, Any]:
    damage_rows = [dmg for event in events for dmg in event["damages"]]
    rejected_rows = [row for row in damage_rows if row["gate_outcome"] == "rejected"]
    accepted_rows = [row for row in damage_rows if row["gate_outcome"] == "accepted"]
    d_values = [row["D_rho_max"] for row in rejected_rows]
    accepted_d_values = [row["D_rho_max"] for row in accepted_rows]
    accepted_a_values = [row["A_axis_max"] for row in accepted_rows]
    d_excesses = [row["D_rho_excess"] for row in rejected_rows if row["D_rho_excess"] is not None]
    axis_excesses = [
        row["A_axis_excess"] for row in rejected_rows if row["A_axis_excess"] is not None
    ]

    reason_counts: Counter[str] = Counter()
    location_counts: Counter[str] = Counter()
    accepted_lambda_counts: Counter[str] = Counter()
    step_source_counts: Counter[str] = Counter()
    damage_outcome_counts: Counter[str] = Counter()
    for event in events:
        step_source_counts[event["step_source"]] += 1
        if event["accepted_lambda"] is not None:
            accepted_lambda_counts[f"{event['accepted_lambda']:.6g}"] += 1
        for dmg in event["damages"]:
            damage_outcome_counts[dmg["gate_outcome"]] += 1
            if dmg["gate_outcome"] != "rejected":
                continue
            for reason in dmg["reasons"]:
                reason_counts[reason] += 1
            location = dmg["location"]
            if location["kind"] == "axis_row":
                location_counts[f"axis_j={location['j']}"] += 1
            else:
                location_counts[location["kind"]] += 1

    false_fire_candidates = sum(1 for row in rejected_rows if row["false_fire_candidate"])
    missed_fire_candidates = sum(1 for row in accepted_rows if row["missed_fire_candidate"])
    events_with_fire = sum(
        1 for event in events if any(row["gate_outcome"] == "rejected" for row in event["damages"])
    )
    events_with_accepted = sum(
        1 for event in events if any(row["gate_outcome"] == "accepted" for row in event["damages"])
    )
    no_fire_events = len(events) - events_with_fire

    summary = {
        "run": str(run_dir),
        "log_path": str(log_path),
        "config": config,
        "thresholds": {
            "remap_damage_dmax": dmax,
            "remap_damage_axis_eta": axis_eta,
        },
        "events_total": len(events),
        "events_with_gate_fire": events_with_fire,
        "events_without_gate_fire": no_fire_events,
        "events_with_damage_accepted": events_with_accepted,
        "events_without_damage_accepted": len(events) - events_with_accepted,
        "damage_fire_lines": len(rejected_rows),
        "damage_rejected_lines": len(rejected_rows),
        "damage_accepted_lines": len(accepted_rows),
        "axis_emergency_bypass_events": sum(1 for event in events if event["axis_emergency_bypass"]),
        "phase9_emergency_bypass_events": sum(1 for event in events if event["phase9_emergency_bypass"]),
        "false_fire_candidates": false_fire_candidates,
        "missed_fire_candidates": missed_fire_candidates,
        "missed_fire_candidates_observable": len(accepted_rows),
        "missed_fire_observability": (
            "observable from damage_accepted logs"
            if accepted_rows
            else "no damage_accepted lines found; input may predate accepted-candidate logging"
        ),
        "D_rho_max_logged_min": min(d_values) if d_values else None,
        "D_rho_max_logged_max": max(d_values) if d_values else None,
        "D_rho_max_accepted_min": min(accepted_d_values) if accepted_d_values else None,
        "D_rho_max_accepted_max": max(accepted_d_values) if accepted_d_values else None,
        "A_axis_accepted_max": max(accepted_a_values) if accepted_a_values else None,
        "D_rho_excess_min": min(d_excesses) if d_excesses else None,
        "D_rho_excess_max": max(d_excesses) if d_excesses else None,
        "A_axis_excess_max": max(axis_excesses) if axis_excesses else None,
        "fire_reason_counts": dict(sorted(reason_counts.items())),
        "fire_location_counts": dict(location_counts.most_common()),
        "damage_outcome_counts": dict(sorted(damage_outcome_counts.items())),
        "accepted_lambda_counts": dict(sorted(accepted_lambda_counts.items())),
        "step_source_counts": dict(sorted(step_source_counts.items())),
    }
    return {"summary": summary, "events": events}


def collect_runs(paths: list[Path]) -> list[tuple[Path, Path]]:
    seen: set[Path] = set()
    runs: list[tuple[Path, Path]] = []
    for path in paths:
        run_dir, log_path = find_log(path)
        key = log_path.resolve()
        if key in seen:
            continue
        seen.add(key)
        runs.append((run_dir, log_path))
    return sorted(runs, key=lambda item: str(item[0]))


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    input_paths = [Path(item) for item in args.paths] if args.paths else default_inputs()
    runs = collect_runs(input_paths)
    parsed = []
    for run_dir, log_path in runs:
        config = load_config(run_dir)
        parsed.append(parse_log(run_dir, log_path, config, args))

    accepted_lines_total = sum(item["summary"]["damage_accepted_lines"] for item in parsed)
    totals = {
        "runs": len(parsed),
        "events_total": sum(item["summary"]["events_total"] for item in parsed),
        "events_with_gate_fire": sum(item["summary"]["events_with_gate_fire"] for item in parsed),
        "events_with_damage_accepted": sum(
            item["summary"]["events_with_damage_accepted"] for item in parsed
        ),
        "damage_fire_lines": sum(item["summary"]["damage_fire_lines"] for item in parsed),
        "damage_rejected_lines": sum(item["summary"]["damage_rejected_lines"] for item in parsed),
        "damage_accepted_lines": accepted_lines_total,
        "axis_emergency_bypass_events": sum(
            item["summary"]["axis_emergency_bypass_events"] for item in parsed
        ),
        "phase9_emergency_bypass_events": sum(
            item["summary"]["phase9_emergency_bypass_events"] for item in parsed
        ),
        "false_fire_candidates": sum(item["summary"]["false_fire_candidates"] for item in parsed),
        "missed_fire_candidates": sum(item["summary"]["missed_fire_candidates"] for item in parsed),
        "missed_fire_candidates_observable": accepted_lines_total,
        "missed_fire_observability": (
            "observable from damage_accepted logs"
            if accepted_lines_total
            else "no damage_accepted lines found; input may predate accepted-candidate logging"
        ),
    }
    report = {"totals": totals, "runs": []}
    for item in parsed:
        if args.events:
            report["runs"].append(item)
        else:
            report["runs"].append({"summary": item["summary"]})
    return report


def print_text(report: dict[str, Any]) -> None:
    totals = report["totals"]
    print("Phase 9 damage gate fire-pattern summary")
    print(
        "runs={runs} events={events_total} events_with_fire={events_with_gate_fire} "
        "fire_lines={damage_fire_lines} events_with_accepted={events_with_damage_accepted} "
        "accepted_lines={damage_accepted_lines} axis_bypass={axis_emergency_bypass_events} "
        "emergency_bypass={phase9_emergency_bypass_events} false_fire_candidates={false_fire_candidates} "
        "missed_fire_candidates={missed_fire_candidates}".format(
            **totals
        )
    )
    print(f"missed_fire_observability={totals['missed_fire_observability']}")
    print()
    print(
        "run,events,events_with_fire,fire_lines,events_with_accepted,accepted_lines,"
        "D_min,D_max,D_accepted_min,D_accepted_max,D_excess_min,D_excess_max,"
        "axis_bypass,emergency_bypass,false_fire,missed_fire,top_locations"
    )
    for item in report["runs"]:
        summary = item["summary"]
        locations = ";".join(
            f"{key}:{value}" for key, value in list(summary["fire_location_counts"].items())[:4]
        )
        print(
            f"{summary['run']},{summary['events_total']},{summary['events_with_gate_fire']},"
            f"{summary['damage_fire_lines']},{summary['events_with_damage_accepted']},"
            f"{summary['damage_accepted_lines']},{summary['D_rho_max_logged_min']},"
            f"{summary['D_rho_max_logged_max']},{summary['D_rho_max_accepted_min']},"
            f"{summary['D_rho_max_accepted_max']},{summary['D_rho_excess_min']},"
            f"{summary['D_rho_excess_max']},{summary['axis_emergency_bypass_events']},"
            f"{summary['phase9_emergency_bypass_events']},{summary['false_fire_candidates']},"
            f"{summary['missed_fire_candidates']},"
            f"{locations}"
        )


def print_csv(report: dict[str, Any]) -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "run",
            "events_total",
            "events_with_gate_fire",
            "events_without_gate_fire",
            "events_with_damage_accepted",
            "events_without_damage_accepted",
            "damage_fire_lines",
            "damage_rejected_lines",
            "damage_accepted_lines",
            "D_rho_max_logged_min",
            "D_rho_max_logged_max",
            "D_rho_max_accepted_min",
            "D_rho_max_accepted_max",
            "A_axis_accepted_max",
            "D_rho_excess_min",
            "D_rho_excess_max",
            "axis_emergency_bypass_events",
            "phase9_emergency_bypass_events",
            "false_fire_candidates",
            "missed_fire_candidates",
            "accepted_lambda_counts",
            "damage_outcome_counts",
            "fire_location_counts",
        ]
    )
    for item in report["runs"]:
        summary = item["summary"]
        writer.writerow(
            [
                summary["run"],
                summary["events_total"],
                summary["events_with_gate_fire"],
                summary["events_without_gate_fire"],
                summary["events_with_damage_accepted"],
                summary["events_without_damage_accepted"],
                summary["damage_fire_lines"],
                summary["damage_rejected_lines"],
                summary["damage_accepted_lines"],
                summary["D_rho_max_logged_min"],
                summary["D_rho_max_logged_max"],
                summary["D_rho_max_accepted_min"],
                summary["D_rho_max_accepted_max"],
                summary["A_axis_accepted_max"],
                summary["D_rho_excess_min"],
                summary["D_rho_excess_max"],
                summary["axis_emergency_bypass_events"],
                summary["phase9_emergency_bypass_events"],
                summary["false_fire_candidates"],
                summary["missed_fire_candidates"],
                json.dumps(summary["accepted_lambda_counts"], sort_keys=True),
                json.dumps(summary["damage_outcome_counts"], sort_keys=True),
                json.dumps(summary["fire_location_counts"], sort_keys=True),
            ]
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan TENRYU logs for Phase 9 ALE remap-damage gate fires. "
            "With no paths, scans build/output_h3b_phase9_11_* and build/output_h3b_256_T31."
        )
    )
    parser.add_argument("paths", nargs="*", help="Run directories or log files to scan")
    parser.add_argument("--format", choices=("text", "json", "csv"), default="text")
    parser.add_argument("--events", action="store_true", help="Include per-event records in JSON output")
    parser.add_argument("--dmax", type=float, default=None, help="Override remap_damage_dmax")
    parser.add_argument("--axis-eta", type=float, default=None, help="Override remap_damage_axis_eta")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.format == "csv":
        print_csv(report)
    else:
        print_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
