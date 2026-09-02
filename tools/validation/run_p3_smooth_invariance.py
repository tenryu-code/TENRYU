#!/usr/bin/env python3
"""Run and gate the P3 polar-tier smooth-flow invariance suite."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CS = math.sqrt((5.0 / 3.0) * 2.0e10 / 1.0)
EPS = 2.220446049250313e-16
VTOL = 1.0e3 * EPS
H = 5.0e4
V0 = 500.0
DT = 5.0e-10
R_DISK = 1.196e-3

VALID_MODES = ("static", "translate", "homologous")
REQUIRED_DATASETS = (
    "mesh/v_r",
    "mesh/v_z",
    "mesh/x_r",
    "mesh/x_z",
    "hydro/rho",
    "hydro/vol",
    "mesh/topology/v3/cell_node_csr_offsets",
    "mesh/topology/v3/cell_node_csr_indices",
)


@dataclass(frozen=True)
class Metric:
    value: float
    gated: bool
    scope: str


@dataclass(frozen=True)
class ModeResult:
    initial_snapshot: Path
    final_snapshot: Path
    final_time_s: float
    metrics: dict[str, Metric]


def parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not modes:
        raise argparse.ArgumentTypeError("--modes must select at least one mode")
    invalid = [mode for mode in modes if mode not in VALID_MODES]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown mode(s) {','.join(invalid)}; choose from {','.join(VALID_MODES)}"
        )
    if len(set(modes)) != len(modes):
        raise argparse.ArgumentTypeError("--modes must not contain duplicates")
    return modes


def scalar_dataset(handle: Any, name: str, default: float = 0.0) -> float:
    if name not in handle:
        return default
    values = np.asarray(handle[name][()], dtype=float).reshape(-1)
    return float(values[-1]) if values.size else default


def snapshot_time_step(snapshot_path: Path) -> tuple[float, int]:
    import h5py

    with h5py.File(snapshot_path, "r") as handle:
        missing = [name for name in REQUIRED_DATASETS if name not in handle]
        if missing:
            raise KeyError(f"{snapshot_path}: missing {', '.join(missing)}")
        time_s = float(handle.attrs.get("t", 0.0))
        time_s = scalar_dataset(handle, "time_state/t", time_s)
        step = int(round(scalar_dataset(handle, "time_state/step", 0.0)))
    return time_s, step


def discover_actual_outdir(outdir: Path) -> Path:
    if (outdir / "run_info.json").exists():
        return outdir
    candidates: list[Path] = []
    if outdir.parent.exists():
        for candidate in outdir.parent.glob(f"{outdir.name}_*"):
            if (candidate / "run_info.json").exists():
                candidates.append(candidate)
    if candidates:
        return max(candidates, key=lambda candidate: candidate.stat().st_mtime)
    return outdir


def first_last_snapshot_paths(outdir: Path) -> tuple[Path, Path]:
    actual = discover_actual_outdir(outdir)
    candidates = sorted(set((actual / "results").glob("*.h5")))
    candidates.extend(path for path in sorted(actual.glob("*.h5")) if path not in candidates)
    if not candidates:
        raise FileNotFoundError(f"no HDF5 snapshots under {actual}")

    readable: list[tuple[int, float, float, Path]] = []
    errors: list[str] = []
    for candidate in candidates:
        try:
            time_s, step = snapshot_time_step(candidate)
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        readable.append((step, time_s, candidate.stat().st_mtime, candidate))
    if not readable:
        detail = "; ".join(errors)
        raise RuntimeError(f"no readable P3 snapshots under {actual}: {detail}")

    first = min(readable, key=lambda item: item[:3])
    last = max(readable, key=lambda item: item[:3])
    if first[0] != 0:
        raise RuntimeError(f"first P3 snapshot under {actual} is step {first[0]}, not 0")
    if first[3] == last[3]:
        raise RuntimeError(f"only one readable P3 snapshot under {actual}")
    return first[3], last[3]


def load_snapshot(snapshot_path: Path) -> dict[str, Any]:
    import h5py

    with h5py.File(snapshot_path, "r") as handle:
        missing = [name for name in REQUIRED_DATASETS if name not in handle]
        if missing:
            raise KeyError(f"{snapshot_path}: missing {', '.join(missing)}")
        time_s = float(handle.attrs.get("t", 0.0))
        time_s = scalar_dataset(handle, "time_state/t", time_s)
        step = int(round(scalar_dataset(handle, "time_state/step", 0.0)))
        snapshot = {
            "time_s": time_s,
            "step": step,
            "v_r": np.asarray(handle["mesh/v_r"][()], dtype=float).reshape(-1),
            "v_z": np.asarray(handle["mesh/v_z"][()], dtype=float).reshape(-1),
            "x_r": np.asarray(handle["mesh/x_r"][()], dtype=float).reshape(-1),
            "x_z": np.asarray(handle["mesh/x_z"][()], dtype=float).reshape(-1),
            "rho": np.asarray(handle["hydro/rho"][()], dtype=float).reshape(-1),
            "vol": np.asarray(handle["hydro/vol"][()], dtype=float).reshape(-1),
            "offsets": np.asarray(
                handle["mesh/topology/v3/cell_node_csr_offsets"][()], dtype=int
            ).reshape(-1),
            "indices": np.asarray(
                handle["mesh/topology/v3/cell_node_csr_indices"][()], dtype=int
            ).reshape(-1),
        }

    node_size = snapshot["x_r"].size
    for name in ("x_z", "v_r", "v_z"):
        if snapshot[name].size != node_size:
            raise ValueError(
                f"{snapshot_path}: {name} has {snapshot[name].size} entries, "
                f"expected {node_size}"
            )
    if snapshot["rho"].size != snapshot["vol"].size:
        raise ValueError(
            f"{snapshot_path}: hydro/rho and hydro/vol sizes differ "
            f"({snapshot['rho'].size} != {snapshot['vol'].size})"
        )
    return snapshot


def validate_snapshot_pair(initial: dict[str, Any], final: dict[str, Any]) -> None:
    for name in ("x_r", "x_z", "v_r", "v_z", "rho", "vol"):
        if initial[name].size != final[name].size:
            raise ValueError(
                f"initial/final {name} sizes differ "
                f"({initial[name].size} != {final[name].size})"
            )


def masked_max(values: np.ndarray, mask: np.ndarray, scale: float) -> float:
    if not np.any(mask):
        raise ValueError("metric mask selects no entries")
    return float(np.max(values[mask]) / scale)


def initial_cell_radii(
    initial: dict[str, Any], node_radius: np.ndarray, node_finite: np.ndarray
) -> np.ndarray:
    offsets = initial["offsets"]
    indices = initial["indices"]
    n_cells = initial["rho"].size
    if offsets.size != n_cells + 1:
        raise ValueError(
            "initial cell-node CSR offset count does not match hydro/rho "
            f"({offsets.size} != {n_cells + 1})"
        )
    if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("initial cell-node CSR offsets are invalid")
    if offsets[-1] != indices.size:
        raise ValueError(
            "initial cell-node CSR terminal offset does not match index count "
            f"({offsets[-1]} != {indices.size})"
        )

    radii = np.full(n_cells, np.nan, dtype=float)
    for cell in range(n_cells):
        begin = int(offsets[cell])
        end = int(offsets[cell + 1])
        nodes = indices[begin:end]
        if nodes.size == 0:
            raise ValueError(f"initial cell {cell} has no CSR nodes")
        if np.any(nodes < 0) or np.any(nodes >= node_radius.size):
            raise ValueError(f"initial cell {cell} has an out-of-range CSR node")
        if np.all(node_finite[nodes]):
            radii[cell] = float(np.mean(node_radius[nodes]))
    return radii


def rho_spread(rho: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        raise ValueError("rho metric mask selects no cells")
    selected = rho[mask]
    return float((np.max(selected) - np.min(selected)) / np.mean(selected))


def analyze_mode(mode: str, outdir: Path) -> ModeResult:
    initial_path, final_path = first_last_snapshot_paths(outdir)
    initial = load_snapshot(initial_path)
    final = load_snapshot(final_path)
    validate_snapshot_pair(initial, final)

    node_finite = np.isfinite(initial["x_r"]) & np.isfinite(initial["x_z"])
    if not np.any(node_finite):
        raise ValueError(f"{initial_path}: no finite initial node coordinates")
    r0 = np.hypot(initial["x_r"], initial["x_z"])
    r_out = float(np.max(r0[node_finite]))
    if not math.isfinite(r_out) or r_out <= 0.0:
        raise ValueError(f"{initial_path}: invalid initial outer radius {r_out}")

    shield_nodes = node_finite & (r0 <= 0.5 * r_out)
    core_nodes = r0 <= 3.0 * R_DISK
    final_time_s = float(final["time_s"])
    metrics: dict[str, Metric] = {}

    if mode == "static":
        velocity_error = np.hypot(final["v_r"], final["v_z"])
        position_error = np.hypot(
            final["x_r"] - initial["x_r"],
            final["x_z"] - initial["x_z"],
        )
        metrics["m_v"] = Metric(
            masked_max(velocity_error, node_finite, CS), True, "all finite nodes"
        )
        metrics["m_x"] = Metric(
            masked_max(position_error, node_finite, r_out), True, "all finite nodes"
        )
    elif mode == "translate":
        velocity_error = np.hypot(final["v_r"], final["v_z"] - V0)
        metrics["m_v"] = Metric(
            masked_max(velocity_error, shield_nodes, CS),
            True,
            "r0 <= 0.5 R_out",
        )
        metrics["m_v_all"] = Metric(
            masked_max(velocity_error, node_finite, CS), False, "all finite nodes"
        )
    else:
        denominator = 1.0 - H * final_time_s
        if denominator <= 0.0:
            raise ValueError(
                f"homologous velocity reference is singular at t_f={final_time_s}"
            )
        v_ref_r = -H * final["x_r"] / denominator
        v_ref_z = -H * final["x_z"] / denominator
        velocity_error = np.hypot(
            final["v_r"] - v_ref_r,
            final["v_z"] - v_ref_z,
        )
        outside_core = shield_nodes & ~core_nodes
        inside_core = shield_nodes & core_nodes
        metrics["m_v_out"] = Metric(
            masked_max(velocity_error, outside_core, CS),
            True,
            "3 R_disk < r0 <= 0.5 R_out",
        )
        metrics["m_v_in"] = Metric(
            masked_max(velocity_error, inside_core, CS),
            False,
            "r0 <= 3 R_disk and r0 <= 0.5 R_out",
        )

        position_scale = 1.0 - H * final_time_s
        x_ref_r = initial["x_r"] * position_scale
        x_ref_z = initial["x_z"] * position_scale
        position_error = np.hypot(
            final["x_r"] - x_ref_r,
            final["x_z"] - x_ref_z,
        )
        metrics["m_x"] = Metric(
            masked_max(position_error, outside_core, r_out),
            True,
            "3 R_disk < r0 <= 0.5 R_out",
        )

        cell_r0 = initial_cell_radii(initial, r0, node_finite)
        shield_cells = np.isfinite(cell_r0) & (cell_r0 <= 0.5 * r_out)
        metrics["m_rho"] = Metric(
            rho_spread(final["rho"], shield_cells),
            True,
            "cell mean-node r0 <= 0.5 R_out",
        )

    return ModeResult(
        initial_snapshot=initial_path,
        final_snapshot=final_path,
        final_time_s=final_time_s,
        metrics=metrics,
    )


def run_mode(args: argparse.Namespace, mode: str, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "TENRYU_P3_MODE": mode,
            "TENRYU_P3_OUTDIR": str(outdir),
        }
    )
    log_path = outdir / "run.log"
    cmd = [str(args.binary), "run", str(args.deck)]
    print(f"[run_p3_smooth_invariance] mode={mode} outdir={outdir}")
    with log_path.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"mode {mode} failed with return code {proc.returncode}; see {log_path}"
        )


def print_results(summary: dict[str, ModeResult]) -> None:
    for mode, result in summary.items():
        print()
        print(
            f"mode={mode} t_f_s={result.final_time_s:.9e} "
            f"initial={result.initial_snapshot} final={result.final_snapshot}"
        )
        print(f"{'metric':<12} {'value':>14} {'status':>12}  scope")
        for name, metric in result.metrics.items():
            status = "HARD" if metric.gated else "REPORT-ONLY"
            print(f"{name:<12} {metric.value:14.6e} {status:>12}  {metric.scope}")


def gate_passes(summary: dict[str, ModeResult]) -> bool:
    passed = True
    print()
    print(f"{'mode':<12} {'metric':<12} {'value':>14} {'limit':>14} {'PASS/FAIL':>9}")
    for mode, result in summary.items():
        for name, metric in result.metrics.items():
            if not metric.gated:
                continue
            row_passed = math.isfinite(metric.value) and metric.value <= VTOL
            passed = passed and row_passed
            verdict = "PASS" if row_passed else "FAIL"
            print(
                f"{mode:<12} {name:<12} {metric.value:14.6e} "
                f"{VTOL:14.6e} {verdict:>9}"
            )
    print("GATE PASS" if passed else "GATE FAIL")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("./build/tenryu"))
    parser.add_argument(
        "--deck",
        type=Path,
        default=Path("examples/verification/2d_rz_polar_tier_smooth_invariance.py"),
    )
    parser.add_argument(
        "--modes",
        type=parse_modes,
        default="static,translate,homologous",
        help="comma-separated subset of static,translate,homologous",
    )
    parser.add_argument("--outdir-root", type=Path, default=Path("./build"))
    parser.add_argument("--gate", action="store_true")
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="analyze existing output directories without launching TENRYU",
    )
    args = parser.parse_args()

    summary: dict[str, ModeResult] = {}
    for mode in args.modes:
        outdir = args.outdir_root / f"output_p3_smooth_{mode}"
        if not args.skip_run:
            run_mode(args, mode, outdir)
        summary[mode] = analyze_mode(mode, outdir)

    print_results(summary)
    if args.gate:
        return 0 if gate_passes(summary) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
