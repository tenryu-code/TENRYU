#!/usr/bin/env python3
"""Hammer-Rosen supersonic Marshak reference checker.

Units for the analytic reference are HeV, ns, cm, MJ/g, and MJ/ns/cm^2.
Run-file inputs are TENRYU HDF5 outputs in cgs+eV.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np


ALPHA = 1.5
BETA = 1.6
MU = 0.14
LAM = 0.2
F_MJ_G = 3.4
G_INV_K = 1.0 / 7200.0
SIGMA_HR = 1.0285e-2
EPS = BETA / (4.0 + ALPHA)
C_BASE = (16.0 / (4.0 + ALPHA)) * (G_INV_K * SIGMA_HR) / (3.0 * F_MJ_G)
RHO0 = 0.2
ZETA_THRESHOLDS = (0.5, 0.1)
SNAPSHOT_DATASETS = ("hydro/Te", "mesh/x_r", "hydro/vol")
OPTIONAL_ION_DATASETS = ("hydro/Ti", "hydro/ei", "hydro/ee", "hydro/rho")


class StopCheck(RuntimeError):
    pass


def stop(message: str) -> None:
    raise StopCheck(message)


def c_of_rho(rho: float) -> float:
    return C_BASE * rho ** (-(2.0 - MU + LAM))


def trapezoid_cumulative(t_ns: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=float)
    if y.size > 1:
        dt = np.diff(t_ns)
        out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * dt)
    return out


def central_dydt(t_ns: np.ndarray, y: np.ndarray) -> np.ndarray:
    dydt = np.zeros_like(y, dtype=float)
    n = y.size
    if n == 1:
        return dydt
    dydt[0] = (y[1] - y[0]) / (t_ns[1] - t_ns[0])
    dydt[-1] = (y[-1] - y[-2]) / (t_ns[-1] - t_ns[-2])
    if n > 2:
        dydt[1:-1] = (y[2:] - y[:-2]) / (t_ns[2:] - t_ns[:-2])
    return dydt


def hammer_rosen_series(t_ns: np.ndarray, surface_T_HeV: np.ndarray, rho: float = RHO0):
    h = surface_T_HeV ** (4.0 + ALPHA)
    cumint_h = trapezoid_cumulative(t_ns, h)
    dhdt = central_dydt(t_ns, h)
    c_rho = c_of_rho(rho)
    prefactor = ((2.0 + EPS) / (1.0 - EPS)) * c_rho
    x_f_sq = prefactor * np.power(h, -EPS) * cumint_h
    x_f = np.sqrt(np.maximum(x_f_sq, 0.0))
    energy = F_MJ_G * rho ** (1.0 - MU) * x_f * np.power(h, EPS) * (1.0 - EPS)
    return h, cumint_h, dhdt, x_f, energy


def zeta_full(y: float, hs: float) -> float:
    base = (1.0 - y) * (1.0 + 0.5 * EPS * (1.0 - hs) * y)
    if base <= 0.0:
        return 0.0
    return base ** (1.0 / (1.0 - EPS))


def zeta_zeroth(y: float) -> float:
    base = max(1.0 - y, 0.0)
    return base ** (1.0 / (1.0 - EPS))


def invert_zeta(zeta_th: float, hs: float, corrected: bool = True) -> float:
    if not (0.0 < zeta_th < 1.0):
        stop("zeta threshold must be between 0 and 1")
    lo = 0.0
    hi = 1.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        val = zeta_full(mid, hs) if corrected else zeta_zeroth(mid)
        if val > zeta_th:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def predicted_contours(h, dhdt, x_f, rho: float = RHO0):
    c_rho = c_of_rho(rho)
    out = {z: np.full_like(x_f, np.nan, dtype=float) for z in ZETA_THRESHOLDS}
    for i in range(x_f.size):
        if not (x_f[i] > 0.0 and h[i] > 0.0):
            continue
        hs = (x_f[i] * x_f[i] / (c_rho * h[i] ** (2.0 - EPS))) * dhdt[i]
        if not math.isfinite(hs):
            continue
        for zeta_th in ZETA_THRESHOLDS:
            out[zeta_th][i] = invert_zeta(zeta_th, hs) * x_f[i]
    return out


def list_datasets(handle: h5py.File) -> list[str]:
    names: list[str] = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            names.append(name)

    handle.visititems(visit)
    return sorted(names)


def require_dataset(handle: h5py.File, path: str, source: Path) -> np.ndarray:
    if path not in handle:
        available = "\n  ".join(list_datasets(handle))
        stop(f"{source}: required dataset '{path}' not found. Available datasets:\n  {available}")
    return np.asarray(handle[path])


def snapshot_paths(run_dir: Path) -> list[Path]:
    files = [
        p
        for p in sorted(run_dir.glob("*.h5"))
        if "_history" not in p.name and "_ckpt_" not in p.name
    ]
    if not files:
        stop(f"No snapshot .h5 files found in {run_dir}")
    return files


def read_snapshot_time(path: Path) -> float:
    with h5py.File(path, "r") as handle:
        if "t" not in handle.attrs:
            stop(f"{path}: missing root attribute 't'")
        return float(handle.attrs["t"])


def validate_snapshot_schema(path: Path) -> None:
    with h5py.File(path, "r") as handle:
        for dataset in SNAPSHOT_DATASETS:
            require_dataset(handle, dataset, path)


def measured_crossing_depth(depth: np.ndarray, te_eV: np.ndarray, threshold_eV: float) -> float:
    if not math.isfinite(threshold_eV):
        return math.nan
    if depth.size != te_eV.size:
        stop("Internal error: depth and Te arrays have different sizes")
    for i in range(depth.size - 1):
        a = te_eV[i] - threshold_eV
        b = te_eV[i + 1] - threshold_eV
        if a == 0.0:
            return float(depth[i])
        if a >= 0.0 and b <= 0.0:
            denom = te_eV[i + 1] - te_eV[i]
            if denom == 0.0:
                return float(depth[i])
            frac = (threshold_eV - te_eV[i]) / denom
            return float(depth[i] + frac * (depth[i + 1] - depth[i]))
    return math.nan


def load_snapshots(run_dir: Path):
    paths = snapshot_paths(run_dir)
    timed = sorted((read_snapshot_time(p), p) for p in paths)
    t_s = np.asarray([item[0] for item in timed], dtype=float)
    paths = [item[1] for item in timed]
    if t_s.size > 1 and np.any(np.diff(t_s) <= 0.0):
        stop("Snapshot times are not strictly increasing after sorting")
    validate_snapshot_schema(paths[0])

    t_surface = []
    measured = {z: [] for z in ZETA_THRESHOLDS}
    face_area = None
    final_ion = {}
    for path_index, path in enumerate(paths):
        with h5py.File(path, "r") as handle:
            te = require_dataset(handle, "hydro/Te", path).astype(float)
            x_r = require_dataset(handle, "mesh/x_r", path).astype(float)
            vol = require_dataset(handle, "hydro/vol", path).astype(float)
            if te.ndim != 1 or x_r.ndim != 1 or vol.ndim != 1:
                stop(f"{path}: expected 1D hydro/Te, mesh/x_r, and hydro/vol")
            if x_r.size != te.size + 1 or vol.size != te.size:
                stop(f"{path}: mesh/x_r, hydro/Te, and hydro/vol sizes are inconsistent")
            centers = 0.5 * (x_r[:-1] + x_r[1:])
            depth = (x_r[-1] - centers)[::-1]
            te_outer_to_inner = te[::-1]
            surface_eV = float(te[-1])
            t_surface.append(surface_eV / 100.0)
            for zeta_th in ZETA_THRESHOLDS:
                threshold = surface_eV * zeta_th ** (1.0 / (4.0 + ALPHA))
                measured[zeta_th].append(
                    measured_crossing_depth(depth, te_outer_to_inner, threshold)
                )
            if path_index == 0:
                dr_outer = x_r[-1] - x_r[-2]
                if not (dr_outer > 0.0):
                    stop(f"{path}: invalid outer-cell dr for face-area calculation")
                face_area = float(vol[-1] / dr_outer)
            if path_index == len(paths) - 1:
                final_ion = ion_isolation_diagnostic(handle, te, vol, path)

    return {
        "paths": paths,
        "t_s": t_s,
        "t_ns": t_s * 1.0e9,
        "T_S_HeV": np.asarray(t_surface, dtype=float),
        "measured_contours": {
            z: np.asarray(values, dtype=float) for z, values in measured.items()
        },
        "face_area_cm2": face_area,
        "ion": final_ion,
    }


def ion_isolation_diagnostic(handle: h5py.File, te, vol, source: Path) -> dict:
    present = {name: (name in handle) for name in OPTIONAL_ION_DATASETS}
    missing = [name for name, ok in present.items() if not ok]
    if missing:
        return {"available": False, "missing": missing}
    ti = np.asarray(handle["hydro/Ti"], dtype=float)
    ei = np.asarray(handle["hydro/ei"], dtype=float)
    ee = np.asarray(handle["hydro/ee"], dtype=float)
    rho = np.asarray(handle["hydro/rho"], dtype=float)
    if not (ti.shape == ei.shape == ee.shape == rho.shape == te.shape == vol.shape):
        stop(f"{source}: hydro/Ti, ei, ee, rho, Te, vol shapes differ")
    mask = te > 50.0
    if not np.any(mask):
        return {"available": True, "n_cells": 0, "max_Ti_eV": math.nan, "ion_fraction": math.nan}
    weights = rho[mask] * vol[mask]
    ion = np.sum(weights * ei[mask])
    total = np.sum(weights * (ee[mask] + ei[mask]))
    frac = float(ion / total) if total != 0.0 else math.nan
    return {
        "available": True,
        "n_cells": int(np.count_nonzero(mask)),
        "max_Ti_eV": float(np.max(ti[mask])),
        "ion_fraction": frac,
    }


def select_history_dataset(handle: h5py.File, preferred: str, legacy: str, tokens: tuple[str, ...]):
    if preferred in handle:
        return preferred
    if legacy in handle:
        return legacy
    datasets = list_datasets(handle)
    candidates = [
        name
        for name in datasets
        if name.startswith("energy/") and all(token in name.lower() for token in tokens)
    ]
    if len(candidates) == 1:
        return candidates[0]
    available = "\n  ".join(datasets)
    stop(
        "Ambiguous or missing history energy series for tokens "
        f"{tokens}. Candidates: {candidates}\nAvailable datasets:\n  {available}"
    )


def history_path(run_dir: Path) -> Path:
    histories = sorted(run_dir.glob("*_history.h5"))
    if len(histories) != 1:
        names = "\n  ".join(str(p) for p in histories) if histories else "(none)"
        stop(f"Expected exactly one history file in {run_dir}; found:\n  {names}")
    return histories[0]


def read_history_energy(run_dir: Path, t_snapshot_s: np.ndarray, face_area_cm2: float):
    path = history_path(run_dir)
    with h5py.File(path, "r") as handle:
        if "t" not in handle:
            stop(f"{path}: missing history dataset 't'")
        t_hist = np.asarray(handle["t"], dtype=float)
        marshak_name = select_history_dataset(
            handle, "energy/marshak_in", "energy/E_Marshak_in", ("marshak",)
        )
        escaped_name = select_history_dataset(
            handle, "energy/radiation_escaped", "energy/E_rad_esc", ("radiation", "escap")
        )
        marshak = np.asarray(handle[marshak_name], dtype=float)
        escaped = np.asarray(handle[escaped_name], dtype=float)
    if not (t_hist.size == marshak.size == escaped.size):
        stop(f"{path}: t, {marshak_name}, and {escaped_name} lengths differ")
    if t_hist.size == 0:
        stop(f"{path}: empty history")
    if t_hist.size > 1 and np.any(np.diff(t_hist) < 0.0):
        stop(f"{path}: history t is not monotone")

    net_cumulative_erg = np.cumsum(marshak - escaped)
    if t_hist[0] > 0.0:
        t_hist = np.concatenate(([0.0], t_hist))
        net_cumulative_erg = np.concatenate(([0.0], net_cumulative_erg))
    energy_mj_cm2 = np.interp(
        t_snapshot_s, t_hist, net_cumulative_erg, left=np.nan, right=np.nan
    )
    energy_mj_cm2 = energy_mj_cm2 * 1.0e-13 / face_area_cm2
    return {
        "path": path,
        "marshak_name": marshak_name,
        "escaped_name": escaped_name,
        "t_hist_s": t_hist,
        "net_cumulative_erg": net_cumulative_erg,
        "energy_MJ_cm2": energy_mj_cm2,
    }


def rel_error(measured: float, predicted: float) -> float:
    if not (math.isfinite(measured) and math.isfinite(predicted)) or predicted == 0.0:
        return math.inf
    return abs(measured - predicted) / abs(predicted)


def nearest_index(t_ns: np.ndarray, target_ns: float) -> int:
    return int(np.argmin(np.abs(t_ns - target_ns)))


def parse_time_list(raw: str, name: str) -> tuple[float, ...]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            stop(f"{name}: empty time entry")
        try:
            value = float(item)
        except ValueError:
            stop(f"{name}: invalid time '{item}'")
        if not math.isfinite(value):
            stop(f"{name}: non-finite time '{item}'")
        values.append(value)
    if not values:
        stop(f"{name}: expected at least one time")
    return tuple(values)


def monotonicity_check(t_ns, measured_contours, times_ns):
    targets = sorted(set(times_ns))
    time_rows = []
    for target in targets:
        idx = nearest_index(t_ns, target)
        time_rows.append({"target_ns": target, "index": idx, "t_ns": float(t_ns[idx])})

    checks = {}
    all_pass = True
    for zeta_th in ZETA_THRESHOLDS:
        values = [float(measured_contours[zeta_th][row["index"]]) for row in time_rows]
        ok = all(math.isfinite(v) for v in values) and all(
            values[i + 1] > values[i] for i in range(len(values) - 1)
        )
        all_pass = all_pass and ok
        checks[str(zeta_th)] = {"measured_cm": values, "pass": ok}

    return {"times": time_rows, "checks": checks, "pass": all_pass}


def run_check(args) -> int:
    run_dir = Path(args.run_dir)
    gate_times = parse_time_list(args.gate_times, "--gate-times")
    report_times = parse_time_list(args.report_times, "--report-times")
    snapshots = load_snapshots(run_dir)
    t_ns = snapshots["t_ns"]
    h, cumint_h, dhdt, x_f, e_pred = hammer_rosen_series(t_ns, snapshots["T_S_HeV"])
    contours_pred = predicted_contours(h, dhdt, x_f)
    history = read_history_energy(run_dir, snapshots["t_s"], snapshots["face_area_cm2"])

    all_pass = True
    rows = []
    time_specs = [(target, "gate", i) for i, target in enumerate(gate_times)]
    time_specs.extend((target, "report", i) for i, target in enumerate(report_times))
    time_specs.sort(key=lambda item: (item[0], 0 if item[1] == "report" else 1, item[2]))
    for target, kind, _ in time_specs:
        idx = nearest_index(t_ns, target)
        row = {
            "target_ns": target,
            "index": idx,
            "t_ns": float(t_ns[idx]),
            "kind": kind,
            "contours": {},
        }
        for zeta_th in ZETA_THRESHOLDS:
            meas = float(snapshots["measured_contours"][zeta_th][idx])
            pred = float(contours_pred[zeta_th][idx])
            err = rel_error(meas, pred)
            ok = err <= args.tol_pos
            if kind == "gate":
                all_pass = all_pass and ok
            row["contours"][str(zeta_th)] = {
                "measured_cm": meas,
                "predicted_cm": pred,
                "rel_error": err,
                "pass": ok if kind == "gate" else None,
            }
        rows.append(row)

    monotone = monotonicity_check(
        t_ns, snapshots["measured_contours"], gate_times + report_times
    )
    all_pass = all_pass and monotone["pass"]

    energy_target = gate_times[-1]
    idx_energy = nearest_index(t_ns, energy_target)
    e_meas = float(history["energy_MJ_cm2"][idx_energy])
    e_expected = float(e_pred[idx_energy])
    e_err = rel_error(e_meas, e_expected)
    e_pass = e_err <= args.tol_energy
    all_pass = all_pass and e_pass

    print("Hammer-Rosen supersonic Marshak reference")
    print(f"Snapshots: {len(snapshots['paths'])} files, t=[{t_ns[0]:.6g}, {t_ns[-1]:.6g}] ns")
    print("Snapshot datasets: hydro/Te, mesh/x_r, hydro/vol")
    print(
        "History series: "
        f"marshak_in={history['marshak_name']}, escaped={history['escaped_name']}"
    )
    print(f"Outer face area from vol/dr: {snapshots['face_area_cm2']:.8e} cm^2")
    print()
    print_ion_diagnostic(snapshots["ion"])
    print()
    print("Position checks:")
    print("target_ns  snap_ns   zeta  measured_cm   predicted_cm  rel_err   status")
    for row in rows:
        for zeta_th in ZETA_THRESHOLDS:
            item = row["contours"][str(zeta_th)]
            status = (
                "trend"
                if row["kind"] == "report"
                else ("PASS" if item["pass"] else "FAIL")
            )
            print(
                f"{row['target_ns']:8.3f}  {row['t_ns']:7.3f}  {zeta_th:4.1f}  "
                f"{item['measured_cm']:12.6e}  {item['predicted_cm']:12.6e}  "
                f"{item['rel_error']:8.3e}  {status}"
            )
    print()
    print("Monotonicity gate:")
    print("zeta  status  measured_cm_by_time")
    for zeta_th in ZETA_THRESHOLDS:
        item = monotone["checks"][str(zeta_th)]
        values = " ".join(f"{v:.6e}" for v in item["measured_cm"])
        print(f"{zeta_th:4.1f}  {'PASS' if item['pass'] else 'FAIL'}    {values}")
    print()
    print(
        f"Energy gate at nearest {energy_target:.6g} ns snapshot "
        f"(t={t_ns[idx_energy]:.6g} ns): measured={e_meas:.6e} MJ/cm^2, "
        f"predicted={e_expected:.6e} MJ/cm^2, rel_err={e_err:.3e}, "
        f"{'PASS' if e_pass else 'FAIL'}"
    )
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")

    if args.json:
        write_json(
            Path(args.json),
            t_ns,
            snapshots,
            h,
            cumint_h,
            dhdt,
            x_f,
            contours_pred,
            e_pred,
            history,
            rows,
            gate_times,
            report_times,
            monotone,
            e_err,
            all_pass,
        )
    return 0 if all_pass else 1


def print_ion_diagnostic(ion: dict) -> None:
    print("Ion-isolation sanity (diagnostic only):")
    if not ion.get("available", False):
        missing = ", ".join(ion.get("missing", []))
        print(f"  unavailable; missing datasets: {missing}")
        return
    print(f"  cells with Te > 50 eV: {ion['n_cells']}")
    print(f"  max Ti over those cells: {ion['max_Ti_eV']:.6e} eV")
    print(f"  ion energy fraction over those cells: {ion['ion_fraction']:.6e}")


def write_json(
    path: Path,
    t_ns,
    snapshots,
    h,
    cumint_h,
    dhdt,
    x_f,
    contours_pred,
    e_pred,
    history,
    rows,
    gate_times,
    report_times,
    monotone,
    e_err,
    all_pass,
) -> None:
    payload = {
        "constants": {
            "alpha": ALPHA,
            "beta": BETA,
            "mu": MU,
            "lambda": LAM,
            "f_MJ_g": F_MJ_G,
            "g_inv_K": G_INV_K,
            "sigma_HR": SIGMA_HR,
            "eps": EPS,
            "rho": RHO0,
            "C": c_of_rho(RHO0),
        },
        "history_series": {
            "marshak_in": history["marshak_name"],
            "radiation_escaped": history["escaped_name"],
        },
        "face_area_cm2": snapshots["face_area_cm2"],
        "t_ns": t_ns.tolist(),
        "T_S_HeV": snapshots["T_S_HeV"].tolist(),
        "H": h.tolist(),
        "cumint_H": cumint_h.tolist(),
        "dH_dt": dhdt.tolist(),
        "x_F_pred_cm": x_f.tolist(),
        "contours_pred_cm": {
            str(z): contours_pred[z].tolist() for z in ZETA_THRESHOLDS
        },
        "contours_meas_cm": {
            str(z): snapshots["measured_contours"][z].tolist() for z in ZETA_THRESHOLDS
        },
        "energy_pred_MJ_cm2": e_pred.tolist(),
        "energy_meas_MJ_cm2": history["energy_MJ_cm2"].tolist(),
        "report_rows": rows,
        "gate_times": list(gate_times),
        "report_times": list(report_times),
        "monotone": monotone,
        "energy_rel_error_3ns": e_err,
        "pass": all_pass,
        "ion_isolation": snapshots["ion"],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def selftest() -> int:
    t_ns = np.linspace(3.0 / 2000.0, 3.0, 2000)
    t0 = 1.5
    k = 1.0 / (4.0 + ALPHA - BETA)
    surface = t0 * t_ns**k
    h, cumint_h, dhdt, x_f, _ = hammer_rosen_series(t_ns, surface)

    h_power = (4.0 + ALPHA) * k
    analytic_cumint = t0 ** (4.0 + ALPHA) * t_ns ** (h_power + 1.0) / (h_power + 1.0)
    _ = analytic_cumint

    c_rho = c_of_rho(RHO0)
    # printed Eq 35 adjudicated a typo (design doc section 1); target is the self-derived Eq-25 closed form
    x_eq25_closed = (
        math.sqrt((2.0 + EPS) / (2.0 - EPS))
        * math.sqrt(c_rho)
        * t0 ** ((4.0 + ALPHA - BETA) / 2.0)
        * t_ns
    )
    x_henyey = (
        (1.0 - EPS) ** -0.5
        * math.sqrt(c_rho)
        * t0 ** ((4.0 + ALPHA - BETA) / 2.0)
        * t_ns
    )
    rel = abs(x_f[-1] - x_eq25_closed[-1]) / abs(x_eq25_closed[-1])
    if rel > 1.0e-4:
        print(f"SELFTEST FAIL: Eq-25 closed-form x_F rel error at 3 ns = {rel:.6e}")
        return 1

    expected_ratio = math.sqrt((2.0 + EPS) * (1.0 - EPS) / (2.0 - EPS))
    ratio = x_f[-1] / x_henyey[-1]
    if abs(ratio - expected_ratio) / abs(expected_ratio) > 1.0e-4:
        print(
            "SELFTEST FAIL: pipeline/Henyey exact-front ratio at 3 ns "
            f"= {ratio:.8f} expected {expected_ratio:.8f}"
        )
        return 1
    print(f"Pipeline/Eq-25 closed-form rel error at 3 ns: {rel:.6e}")
    print(
        "Pipeline/Henyey exact-front ratio documented O(eps^2) theory offset "
        f"(-2.51%): {ratio:.8f} (expected {expected_ratio:.8f})"
    )

    for zeta_th in ZETA_THRESHOLDS:
        y = invert_zeta(zeta_th, hs=0.0, corrected=True)
        z_round = zeta_full(y, hs=0.0)
        if abs(z_round - zeta_th) > 1.0e-12:
            print(f"SELFTEST FAIL: Eq30 round-trip zeta={zeta_th} got {z_round}")
            return 1
        y0 = 1.0 - zeta_th ** (1.0 - EPS)
        if not (y > y0):
            print(f"SELFTEST FAIL: Eq30 contour direction zeta={zeta_th} y={y} y0={y0}")
            return 1
        y0_bisect = invert_zeta(zeta_th, hs=0.0, corrected=False)
        if abs(y0_bisect - y0) > 1.0e-12:
            print(
                "SELFTEST FAIL: zeroth-order profile identity "
                f"zeta={zeta_th} y={y0_bisect} expected={y0}"
            )
            return 1
    print("G0 selftest PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/hammer_rosen_supersonic")
    parser.add_argument("--json", default=None)
    parser.add_argument("--gate-times", default="3.0")
    parser.add_argument("--report-times", default="1.5,2.0")
    parser.add_argument("--tol-pos", type=float, default=0.05)
    parser.add_argument("--tol-energy", type=float, default=0.05)
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.selftest:
            return selftest()
        return run_check(args)
    except StopCheck as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
