#!/usr/bin/env python3

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import h5py
import numpy as np


PROTON_MASS_G = 1.67262192369e-24


@dataclass
class MaterialInfo:
    name: str
    z: np.ndarray
    a_amu: np.ndarray
    mass_fraction: np.ndarray
    number_fraction: np.ndarray | None
    abar_ion_amu: float


@dataclass
class EOSData:
    ni_grid_cm3: np.ndarray
    rho_grid_gcc: np.ndarray
    temperature_grid_eV: np.ndarray
    pe: np.ndarray
    pi: np.ndarray
    ee: np.ndarray
    ei: np.ndarray


@dataclass
class TmatData:
    path: Path
    material: MaterialInfo
    eos: EOSData


def decode_scalar_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def load_tmat(path: Path) -> TmatData:
    with h5py.File(path, "r") as f:
        name = decode_scalar_string(f["/material/name"][()])
        z = np.asarray(f["/material/Z"][()], dtype=np.int32)
        a_amu = np.asarray(f["/material/A_amu"][()], dtype=np.float64)
        mass_fraction = np.asarray(f["/material/mass_fraction"][()], dtype=np.float64)
        number_fraction = None
        if "/material/number_fraction" in f:
            number_fraction = np.asarray(
                f["/material/number_fraction"][()], dtype=np.float64
            )
        abar_ion_amu = float(f["/material/Abar_ion_amu"][()])

        ni_grid_cm3 = np.asarray(f["/eos/grid/ni_cm3"][()], dtype=np.float64)
        temperature_grid_eV = np.asarray(
            f["/eos/grid/temperature_eV"][()], dtype=np.float64
        )
        rho_grid_gcc = ni_grid_cm3 * abar_ion_amu * PROTON_MASS_G

        pe = np.asarray(f["/eos/fields/P_e"][()], dtype=np.float64)
        pi = np.asarray(f["/eos/fields/P_i"][()], dtype=np.float64)
        ee = np.asarray(f["/eos/fields/e_e"][()], dtype=np.float64)
        ei = np.asarray(f["/eos/fields/e_i"][()], dtype=np.float64)

    material = MaterialInfo(
        name=name,
        z=z,
        a_amu=a_amu,
        mass_fraction=mass_fraction,
        number_fraction=number_fraction,
        abar_ion_amu=abar_ion_amu,
    )
    eos = EOSData(
        ni_grid_cm3=ni_grid_cm3,
        rho_grid_gcc=rho_grid_gcc,
        temperature_grid_eV=temperature_grid_eV,
        pe=pe,
        pi=pi,
        ee=ee,
        ei=ei,
    )
    return TmatData(path=path, material=material, eos=eos)


def bracket_index(log_grid: np.ndarray, x: float) -> tuple[int, int]:
    if log_grid.size < 2:
        return 0, 0
    if x <= log_grid[0]:
        return 0, 1
    if x >= log_grid[-1]:
        return log_grid.size - 2, log_grid.size - 1
    hi = int(np.searchsorted(log_grid, x, side="right"))
    return hi - 1, hi


def bilinear_log_interp(
    rho_grid: np.ndarray,
    t_grid: np.ndarray,
    values_dt: np.ndarray,
    rho_gcc: float,
    temperature_eV: float,
) -> float:
    rho_safe = float(np.clip(rho_gcc, rho_grid[0], rho_grid[-1]))
    t_safe = float(np.clip(temperature_eV, t_grid[0], t_grid[-1]))

    log_rho_grid = np.log(rho_grid)
    log_t_grid = np.log(t_grid)
    x = math.log(rho_safe)
    y = math.log(t_safe)

    i0, i1 = bracket_index(log_rho_grid, x)
    j0, j1 = bracket_index(log_t_grid, y)

    x0 = log_rho_grid[i0]
    x1 = log_rho_grid[i1]
    y0 = log_t_grid[j0]
    y1 = log_t_grid[j1]

    tx = 0.0 if x1 <= x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 <= y0 else (y - y0) / (y1 - y0)

    v00 = values_dt[i0, j0]
    v10 = values_dt[i1, j0]
    v01 = values_dt[i0, j1]
    v11 = values_dt[i1, j1]

    vx0 = v00 + tx * (v10 - v00)
    vx1 = v01 + tx * (v11 - v01)
    return float(vx0 + ty * (vx1 - vx0))


def evaluate_profile(eos: EOSData, rho_gcc: float, temperatures_eV: np.ndarray) -> dict[str, np.ndarray]:
    pe = np.array(
        [
            bilinear_log_interp(
                eos.rho_grid_gcc, eos.temperature_grid_eV, eos.pe, rho_gcc, t
            )
            for t in temperatures_eV
        ],
        dtype=np.float64,
    )
    pi = np.array(
        [
            bilinear_log_interp(
                eos.rho_grid_gcc, eos.temperature_grid_eV, eos.pi, rho_gcc, t
            )
            for t in temperatures_eV
        ],
        dtype=np.float64,
    )
    ee = np.array(
        [
            bilinear_log_interp(
                eos.rho_grid_gcc, eos.temperature_grid_eV, eos.ee, rho_gcc, t
            )
            for t in temperatures_eV
        ],
        dtype=np.float64,
    )
    ei = np.array(
        [
            bilinear_log_interp(
                eos.rho_grid_gcc, eos.temperature_grid_eV, eos.ei, rho_gcc, t
            )
            for t in temperatures_eV
        ],
        dtype=np.float64,
    )
    ratio = np.divide(pe, pi, out=np.full_like(pe, np.nan), where=(pi != 0.0))
    return {"Pe": pe, "Pi": pi, "ee": ee, "ei": ei, "Pe_over_Pi": ratio}


def root_bisect(
    func: Callable[[float], float], lo: float, hi: float, iterations: int = 80
) -> float:
    flo = func(lo)
    fhi = func(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        return 0.5 * (lo + hi)

    a = lo
    b = hi
    fa = flo
    for _ in range(iterations):
        mid = 0.5 * (a + b)
        fm = func(mid)
        if fm == 0.0:
            return mid
        if fa * fm <= 0.0:
            b = mid
        else:
            a = mid
            fa = fm
    return 0.5 * (a + b)


def negative_intervals(
    temperatures_eV: np.ndarray, values: np.ndarray
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    neg = values < 0.0
    if not np.any(neg):
        return intervals

    start = None
    for i, is_neg in enumerate(neg):
        if is_neg and start is None:
            start = i
        if start is not None and (i == neg.size - 1 or not neg[i + 1]):
            end = i
            intervals.append(
                (float(temperatures_eV[start]), float(temperatures_eV[end]))
            )
            start = None
    return intervals


def refine_negative_intervals(
    temperatures_eV: np.ndarray,
    values: np.ndarray,
    evaluator: Callable[[float], float],
) -> list[tuple[float, float]]:
    coarse = negative_intervals(temperatures_eV, values)
    if not coarse:
        return []

    refined: list[tuple[float, float]] = []
    neg = values < 0.0
    for t_lo, t_hi in coarse:
        i_lo = int(np.searchsorted(temperatures_eV, t_lo, side="left"))
        i_hi = int(np.searchsorted(temperatures_eV, t_hi, side="right")) - 1

        left = t_lo
        if i_lo > 0 and not neg[i_lo - 1] and neg[i_lo]:
            left = root_bisect(evaluator, float(temperatures_eV[i_lo - 1]), t_lo)

        right = t_hi
        if i_hi + 1 < neg.size and neg[i_hi] and not neg[i_hi + 1]:
            right = root_bisect(evaluator, t_hi, float(temperatures_eV[i_hi + 1]))

        refined.append((left, right))
    return refined


def array_block(values: np.ndarray, fmt: str = "{:.6e}", per_line: int = 8) -> str:
    lines: list[str] = []
    for start in range(0, values.size, per_line):
        chunk = values[start : start + per_line]
        lines.append("  " + " ".join(fmt.format(float(v)) for v in chunk))
    return "\n".join(lines)


def make_profile_temperatures(
    t_grid: np.ndarray, t_min: float, t_max: float
) -> np.ndarray:
    temps = [t_min]
    temps.extend(
        float(t)
        for t in t_grid
        if t_min <= float(t) <= t_max and not math.isclose(float(t), t_min)
    )
    if not math.isclose(temps[-1], t_max):
        temps.append(t_max)
    return np.array(temps, dtype=np.float64)


def print_profile_table(
    eos: EOSData,
    rho_target_gcc: float,
    temperatures_eV: np.ndarray,
    profile: dict[str, np.ndarray],
) -> None:
    t_floor = float(eos.temperature_grid_eV[0])
    print(
        "T_requested_eV,T_eval_eV,Pe_dyne_cm2,Pi_dyne_cm2,Pe_over_Pi,"
        "ee_erg_g,ei_erg_g"
    )
    for i, t_req in enumerate(temperatures_eV):
        t_eval = min(max(float(t_req), t_floor), float(eos.temperature_grid_eV[-1]))
        print(
            f"{t_req:.6e},{t_eval:.6e},{profile['Pe'][i]:.6e},{profile['Pi'][i]:.6e},"
            f"{profile['Pe_over_Pi'][i]:.6e},{profile['ee'][i]:.6e},{profile['ei'][i]:.6e}"
        )


def write_csv(
    out_path: Path,
    eos: EOSData,
    rho_targets_gcc: list[float],
    sample_temperatures_eV: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "rho_target_gcc",
                "T_requested_eV",
                "T_eval_eV",
                "Pe_dyne_cm2",
                "Pi_dyne_cm2",
                "Pe_over_Pi",
                "ee_erg_g",
                "ei_erg_g",
            ]
        )
        t_floor = float(eos.temperature_grid_eV[0])
        t_cap = float(eos.temperature_grid_eV[-1])
        for rho_target_gcc in rho_targets_gcc:
            profile = evaluate_profile(eos, rho_target_gcc, sample_temperatures_eV)
            for i, t_req in enumerate(sample_temperatures_eV):
                writer.writerow(
                    [
                        f"{rho_target_gcc:.8e}",
                        f"{float(t_req):.8e}",
                        f"{min(max(float(t_req), t_floor), t_cap):.8e}",
                        f"{profile['Pe'][i]:.8e}",
                        f"{profile['Pi'][i]:.8e}",
                        f"{profile['Pe_over_Pi'][i]:.8e}",
                        f"{profile['ee'][i]:.8e}",
                        f"{profile['ei'][i]:.8e}",
                    ]
                )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect low-temperature Pe/Pi behavior in a TENRYU TMAT-H5 EOS table "
            "using TENRYU-compatible clamp + log-log bilinear interpolation."
        )
    )
    parser.add_argument(
        "tmat_file",
        nargs="?",
        default="TMAT-H5/CD.tmat.h5",
        help="Path to TMAT-H5 file",
    )
    parser.add_argument(
        "--rho-target",
        dest="rho_targets",
        type=float,
        action="append",
        help="Target mass density in g/cc; may be repeated",
    )
    parser.add_argument("--t-min", type=float, default=0.025)
    parser.add_argument("--t-max", type=float, default=10.0)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=800,
        help="Number of log-spaced temperatures for interval detection / CSV export",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path for dense sampled profiles",
    )
    args = parser.parse_args()

    rho_targets = args.rho_targets if args.rho_targets else [1.0, 0.01]
    tmat = load_tmat(Path(args.tmat_file))
    eos = tmat.eos

    if args.t_min <= 0.0 or args.t_max <= 0.0 or args.t_min > args.t_max:
        raise SystemExit("Temperature range must satisfy 0 < t_min <= t_max")
    if args.n_samples < 2:
        raise SystemExit("--n-samples must be >= 2")

    dense_t = np.geomspace(args.t_min, args.t_max, args.n_samples)
    profile_t = make_profile_temperatures(
        eos.temperature_grid_eV, args.t_min, args.t_max
    )

    print(f"TMAT file: {tmat.path}")
    print(f"Material: {tmat.material.name}")
    print(f"Abar_ion_amu: {tmat.material.abar_ion_amu:.6f}")
    print(
        "Species Z: "
        + " ".join(str(int(v)) for v in np.asarray(tmat.material.z).ravel())
    )
    print(
        "Species A_amu: "
        + " ".join(f"{float(v):.6f}" for v in np.asarray(tmat.material.a_amu).ravel())
    )
    print(
        "Mass fraction: "
        + " ".join(
            f"{float(v):.6f}" for v in np.asarray(tmat.material.mass_fraction).ravel()
        )
    )
    if tmat.material.number_fraction is not None:
        print(
            "Number fraction: "
            + " ".join(
                f"{float(v):.6f}"
                for v in np.asarray(tmat.material.number_fraction).ravel()
            )
        )

    print()
    print(
        "Temperature grid for both electron and ion tables [eV] "
        f"(n={eos.temperature_grid_eV.size}):"
    )
    print(array_block(eos.temperature_grid_eV, fmt="{:.6g}"))
    print()
    print(f"Ion-number-density grid [cm^-3] (n={eos.ni_grid_cm3.size}):")
    print(array_block(eos.ni_grid_cm3, fmt="{:.6e}"))
    print()
    print(f"Mass-density grid converted with Abar*mp [g/cc] (n={eos.rho_grid_gcc.size}):")
    print(array_block(eos.rho_grid_gcc, fmt="{:.6e}"))
    print()

    if args.t_min < float(eos.temperature_grid_eV[0]):
        print(
            f"Note: requested t_min={args.t_min:.6g} eV is below table floor "
            f"{float(eos.temperature_grid_eV[0]):.6g} eV; values are clamped to the "
            "table floor, matching TENRYU runtime interpolation."
        )
        print()

    for rho_target in rho_targets:
        nearest_idx = int(np.argmin(np.abs(eos.rho_grid_gcc - rho_target)))
        rho_nearest = float(eos.rho_grid_gcc[nearest_idx])
        ni_nearest = float(eos.ni_grid_cm3[nearest_idx])
        print("=" * 88)
        print(f"Density target: {rho_target:.6e} g/cc")
        print(
            f"Nearest EOS density grid point: idx={nearest_idx}, "
            f"rho={rho_nearest:.6e} g/cc, ni={ni_nearest:.6e} cm^-3"
        )

        dense_profile = evaluate_profile(eos, rho_target, dense_t)
        pe_eval = lambda temp: bilinear_log_interp(
            eos.rho_grid_gcc, eos.temperature_grid_eV, eos.pe, rho_target, temp
        )
        neg_ranges = refine_negative_intervals(dense_t, dense_profile["Pe"], pe_eval)
        if neg_ranges:
            print("Pe < 0 intervals [eV]:")
            for lo, hi in neg_ranges:
                print(f"  {lo:.6e} .. {hi:.6e}")
        else:
            print("Pe < 0 intervals [eV]: none in requested temperature range")

        print()
        print_profile_table(
            eos, rho_target, profile_t, evaluate_profile(eos, rho_target, profile_t)
        )
        print()

    if args.csv is not None:
        write_csv(args.csv, eos, rho_targets, dense_t)
        print(f"Wrote CSV: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
