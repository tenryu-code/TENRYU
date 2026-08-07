#!/usr/bin/env python3
"""Extract per-stage complex gain from PR G2 radial Fourier audit v2 HDF5."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


BASE = "/diagnostics/radial_fourier_audit_v2/v1"
EPS = 1.0e-300


def unwrap_angle(angle: float) -> float:
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    while angle > math.pi:
        angle -= 2.0 * math.pi
    return angle


def complex_gain(before: dict, after: dict, eps: float = EPS) -> dict:
    cb = complex(before["cre_vol"], before["cim_vol"])
    ca = complex(after["cre_vol"], after["cim_vol"])
    amp_b = abs(cb)
    amp_a = abs(ca)
    dlog_amp = math.log(max(amp_a, eps)) - math.log(max(amp_b, eps))
    phase_b = math.atan2(cb.imag, cb.real)
    phase_a = math.atan2(ca.imag, ca.real)
    dphase = unwrap_angle(phase_a - phase_b)
    if amp_b > eps:
        g = ca / cb
        in_phase = ((ca - cb) * cb.conjugate()).real / (amp_b * amp_b)
    else:
        g = complex(float("nan"), float("nan"))
        in_phase = float("nan")
    return {
        "gain_abs": abs(g),
        "gain_real": g.real,
        "gain_imag": g.imag,
        "dlog_amp": dlog_amp,
        "alpha_per_s": dlog_amp / max(float(after["dt_cycle"]), eps),
        "dphase": dphase,
        "arg_g": math.atan2(g.imag, g.real),
        "amp_before": amp_b,
        "amp_after": amp_a,
        "alpha_in_phase": in_phase,
    }


def load_records(path: Path) -> list[dict]:
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit("SKIPPED: h5py is not installed") from exc

    keys = [
        "cycle", "t_s", "dt_cycle", "stage_id", "stage_phase", "field_id",
        "m", "j", "cre_vol", "cim_vol",
    ]
    with h5py.File(path, "r") as handle:
        if BASE not in handle:
            raise SystemExit(f"missing HDF5 group: {BASE}")
        group = handle[BASE]
        missing = [key for key in keys if key not in group]
        if missing:
            raise SystemExit("missing v2 dataset(s): " + ", ".join(missing))
        arrays = {key: group[key][...] for key in keys}

    n_rows = len(arrays["cycle"])
    for key, values in arrays.items():
        if len(values) != n_rows:
            raise SystemExit(f"dataset length mismatch for {key}")

    records = []
    for idx in range(n_rows):
        records.append({key: arrays[key][idx].item() for key in keys})
    return records


def pair_gains(records: list[dict]) -> list[dict]:
    before_by_key = {}
    rows = []
    for rec in records:
        key = (
            int(rec["cycle"]),
            int(rec["stage_id"]),
            int(rec["field_id"]),
            int(rec["m"]),
            int(rec["j"]),
        )
        phase = int(rec["stage_phase"])
        if phase == 0:
            before_by_key[key] = rec
        elif phase == 1 and key in before_by_key:
            before = before_by_key[key]
            gain = complex_gain(before, rec)
            rows.append({
                "cycle": int(rec["cycle"]),
                "t": float(rec["t_s"]),
                "dt": float(rec["dt_cycle"]),
                "stage_id": int(rec["stage_id"]),
                "field_id": int(rec["field_id"]),
                "m": int(rec["m"]),
                "j": int(rec["j"]),
                "|C_before|": gain["amp_before"],
                "|C_after|": gain["amp_after"],
                "dlog_abs_C": gain["dlog_amp"],
                "|g|": gain["gain_abs"],
                "gain_real": gain["gain_real"],
                "gain_imag": gain["gain_imag"],
                "arg_g": gain["arg_g"],
                "dphase": gain["dphase"],
                "alpha_per_s": gain["alpha_per_s"],
                "alpha_in_phase": gain["alpha_in_phase"],
            })
    rows.sort(key=lambda row: row["dlog_abs_C"], reverse=True)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    by_cycle = defaultdict(list)
    for row in rows:
        by_cycle[row["cycle"]].append(row)

    summaries = []
    for cycle in sorted(by_cycle):
        cycle_rows = by_cycle[cycle]
        top = max(cycle_rows, key=lambda row: row["dlog_abs_C"])
        stage_positive = defaultdict(float)
        for row in cycle_rows:
            stage_positive[row["stage_id"]] += max(row["dlog_abs_C"], 0.0)
        positive_total = sum(stage_positive.values())
        dominant_stage = top["stage_id"]
        if positive_total > 0.0:
            dominant_stage = max(stage_positive, key=stage_positive.get)
            dominant_fraction = stage_positive[dominant_stage] / positive_total
        else:
            dominant_fraction = 0.0
        summaries.append({
            "cycle": cycle,
            "t": top["t"],
            "dt": top["dt"],
            "dominant_stage_id": dominant_stage,
            "max_dlog_abs_C": top["dlog_abs_C"],
            "dominant_field_id": top["field_id"],
            "dominant_m": top["m"],
            "dominant_j": top["j"],
            "dominant_fraction_positive_dlog": dominant_fraction,
            "dominance": "single_stage" if dominant_fraction >= 0.5 else "distributed",
            "alpha_in_phase": top["alpha_in_phase"],
        })

    if summaries:
        counts = Counter(row["dominant_stage_id"] for row in summaries)
        stage, count = counts.most_common(1)[0]
        consistency = str(stage) if count / len(summaries) >= 0.8 else "distributed"
        for row in summaries:
            row["global_stage_consistency"] = consistency
    return summaries


def write_csv(path: Path | None, rows: list[dict], fieldnames: list[str], stream) -> None:
    if path is None:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history_h5", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = pair_gains(load_records(args.history_h5))
    detail_fields = [
        "cycle", "t", "dt", "stage_id", "field_id", "m", "j",
        "|C_before|", "|C_after|", "dlog_abs_C", "|g|", "gain_real",
        "gain_imag", "arg_g", "dphase", "alpha_per_s", "alpha_in_phase",
    ]
    write_csv(args.output, rows, detail_fields, sys.stdout)

    summary_rows = summarize(rows)
    summary_fields = [
        "cycle", "t", "dt", "dominant_stage_id", "max_dlog_abs_C",
        "dominant_field_id", "dominant_m", "dominant_j",
        "dominant_fraction_positive_dlog", "dominance", "alpha_in_phase",
        "global_stage_consistency",
    ]
    summary_output = args.summary_output
    if summary_output is None and args.output is not None:
        summary_output = args.output.with_suffix(args.output.suffix + ".summary.csv")
    write_csv(summary_output, summary_rows, summary_fields, sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
