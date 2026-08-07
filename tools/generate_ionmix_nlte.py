#!/usr/bin/env python3
"""Generate synthetic IONMIX .cn4 tables for M17 Non-LTE tests."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Callable

import numpy as np

EV_TO_J = 1.6022e-19


def write_record(file_obj, data_bytes: bytes) -> None:
    """Write one Fortran unformatted sequential record."""
    length = len(data_bytes)
    file_obj.write(struct.pack("<i", length))
    file_obj.write(data_bytes)
    file_obj.write(struct.pack("<i", length))


def write_doubles(file_obj, values: np.ndarray) -> None:
    """Write a float64 array as one record."""
    arr = np.asarray(values, dtype=np.float64).ravel(order="C")
    if arr.size == 0:
        write_record(file_obj, b"")
        return
    write_record(file_obj, struct.pack(f"<{arr.size}d", *arr.tolist()))


def write_double(file_obj, value: float) -> None:
    """Write one float64 scalar as one record."""
    write_record(file_obj, struct.pack("<d", float(value)))


def make_opacity_table(
    ntemp: int,
    ndens: int,
    ngroups: int,
    generator: float | Callable[[int, int, int, float, float], float],
    temps_eV: np.ndarray,
    numdens_cm3: np.ndarray,
) -> np.ndarray:
    table = np.empty((ngroups, ndens, ntemp), dtype=np.float64)
    if isinstance(generator, (int, float)):
        table.fill(float(generator))
        return table

    for g in range(ngroups):
        for d in range(ndens):
            ni = float(numdens_cm3[d])
            for t in range(ntemp):
                Te = float(temps_eV[t])
                table[g, d, t] = float(generator(g, d, t, Te, ni))
    return table


def write_ionmix_cn4(
    output_path: Path,
    temps_eV: np.ndarray,
    numdens_cm3: np.ndarray,
    bounds_eV: np.ndarray,
    kappa_R: np.ndarray,
    kappa_PA: np.ndarray,
    kappa_PE: np.ndarray | None,
    eos_blocks: list[np.ndarray] | None = None,
    z_values: np.ndarray | None = None,
    fractions: np.ndarray | None = None,
) -> None:
    ntemp = int(temps_eV.size)
    ndens = int(numdens_cm3.size)
    ngroups = int(bounds_eV.size - 1)

    if eos_blocks is None:
        zbar = np.full((ndens, ntemp), 1.0, dtype=np.float64)
        dpdt_zero = np.zeros((ndens, ntemp), dtype=np.float64)
        pressure_i = np.full((ndens, ntemp), 1.0e8, dtype=np.float64)
        pressure_e = np.full((ndens, ntemp), 1.0e8, dtype=np.float64)
        energy_i = np.full((ndens, ntemp), 1.0e12, dtype=np.float64)
        energy_e = np.full((ndens, ntemp), 1.0e12, dtype=np.float64)
        eos_blocks = [
            zbar,  # block 1: Zbar
            dpdt_zero,  # block 2: dZ/dT (unused)
            pressure_i,  # block 3: P_i
            pressure_e,  # block 4: P_e
            dpdt_zero,  # block 5: dP_i/dT (unused)
            dpdt_zero,  # block 6: dP_e/dT (unused)
            energy_i,  # block 7: e_i
            energy_e,  # block 8: e_e
            dpdt_zero,  # block 9: derivative (unused)
            dpdt_zero,  # block 10: derivative (unused)
            dpdt_zero,  # block 11: derivative (unused)
            dpdt_zero,  # block 12: derivative (unused)
        ]
    if len(eos_blocks) != 12:
        raise ValueError("eos_blocks must contain 12 blocks")

    canonical_blocks = []
    for block in eos_blocks:
        arr = np.asarray(block, dtype=np.float64)
        if arr.shape != (ndens, ntemp):
            raise ValueError(
                f"each EOS block must have shape ({ndens}, {ntemp}), got {arr.shape}"
            )
        canonical_blocks.append(arr)
    eos_blocks = canonical_blocks

    z_values_arr = (
        np.asarray([1.0], dtype=np.float64)
        if z_values is None
        else np.asarray(z_values, dtype=np.float64)
    )
    fractions_arr = (
        np.asarray([1.0], dtype=np.float64)
        if fractions is None
        else np.asarray(fractions, dtype=np.float64)
    )
    if z_values_arr.size != fractions_arr.size or z_values_arr.size == 0:
        raise ValueError("z_values and fractions must have same non-zero length")

    kappa_R_arr = np.asarray(kappa_R, dtype=np.float64)
    kappa_PA_arr = np.asarray(kappa_PA, dtype=np.float64)
    kappa_PE_arr = None if kappa_PE is None else np.asarray(kappa_PE, dtype=np.float64)
    expected_shape = (ngroups, ndens, ntemp)
    if kappa_R_arr.shape != expected_shape:
        raise ValueError(f"kappa_R shape {kappa_R_arr.shape} != expected {expected_shape}")
    if kappa_PA_arr.shape != expected_shape:
        raise ValueError(f"kappa_PA shape {kappa_PA_arr.shape} != expected {expected_shape}")
    if kappa_PE_arr is not None and kappa_PE_arr.shape != expected_shape:
        raise ValueError(f"kappa_PE shape {kappa_PE_arr.shape} != expected {expected_shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        # Header scalars are stored as float64 in IONMIX .cn4.
        write_double(f, float(ntemp))
        write_double(f, float(ndens))

        # Variable-length composition records.
        write_doubles(f, z_values_arr)
        write_doubles(f, fractions_arr)

        write_double(f, float(ngroups))
        write_doubles(f, temps_eV)
        write_doubles(f, numdens_cm3)

        # 12 EOS blocks, each [ndens, ntemp] with T fastest.
        for block in eos_blocks:
            write_doubles(f, block)

        write_doubles(f, bounds_eV)
        write_doubles(f, kappa_R_arr)
        write_doubles(f, kappa_PA_arr)
        if kappa_PE_arr is not None:
            write_doubles(f, kappa_PE_arr)


def generate_eos_idealgas(output_dir: Path) -> Path:
    """Generate binary IONMIX data with ideal-gas-like EOS and constant tabular Zbar."""
    A = 27.0
    Z = 13.0
    zbar_const = 5.0
    ntemp = 10
    ndens = 5

    temps_eV = np.logspace(0.0, 3.0, ntemp, dtype=np.float64)
    numdens_cm3 = np.logspace(20.0, 24.0, ndens, dtype=np.float64)

    ni = numdens_cm3[:, None]
    Te = temps_eV[None, :]
    zbar = np.full((ndens, ntemp), zbar_const, dtype=np.float64)
    zeros = np.zeros((ndens, ntemp), dtype=np.float64)

    pressure_i = ni * Te * EV_TO_J
    pressure_e = zbar * pressure_i

    mass_per_ion_g = A * 1.6726219e-24
    energy_i_1d = 1.5 * temps_eV * EV_TO_J / mass_per_ion_g
    energy_i = np.repeat(energy_i_1d[None, :], ndens, axis=0)
    energy_e = zbar * energy_i

    eos_blocks = [
        zbar,  # block 1: Zbar
        zeros,  # block 2: dZ/dT
        pressure_i,  # block 3: P_i [J/cm^3]
        pressure_e,  # block 4: P_e [J/cm^3]
        zeros,  # block 5: dP_i/dT
        zeros,  # block 6: dP_e/dT
        energy_i,  # block 7: e_i [J/g]
        energy_e,  # block 8: e_e [J/g]
        zeros,  # block 9
        zeros,  # block 10
        zeros,  # block 11
        zeros,  # block 12
    ]

    bounds_eV = np.array([0.1, 100.0], dtype=np.float64)
    kappa_dummy = np.ones((1, ndens, ntemp), dtype=np.float64)
    output_path = output_dir / "ionmix_eos_idealgas.cn4"
    write_ionmix_cn4(
        output_path=output_path,
        temps_eV=temps_eV,
        numdens_cm3=numdens_cm3,
        bounds_eV=bounds_eV,
        kappa_R=kappa_dummy,
        kappa_PA=kappa_dummy,
        kappa_PE=kappa_dummy,
        eos_blocks=eos_blocks,
        z_values=np.array([Z], dtype=np.float64),
        fractions=np.array([1.0], dtype=np.float64),
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic IONMIX .cn4 files for TENRYU Non-LTE tests."
    )
    parser.add_argument("--ntemp", type=int, default=10)
    parser.add_argument("--ndens", type=int, default=5)
    parser.add_argument("--ngroups", type=int, default=4)
    parser.add_argument("--t-min", type=float, default=1.0e-2, dest="t_min")
    # Default t-max=1e4 must match committed test fixtures in tests/data/.
    # Changing this default will alter interpolation behavior for verification tests.
    parser.add_argument("--t-max", type=float, default=1.0e4, dest="t_max")
    parser.add_argument("--ni-min", type=float, default=1.0e18, dest="ni_min")
    parser.add_argument("--ni-max", type=float, default=1.0e24, dest="ni_max")
    parser.add_argument(
        "--bounds",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 5.0, 20.0, 100.0],
        help="Group boundaries in eV (must have ngroups+1 values).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/data"),
        help="Output directory for generated .cn4 files.",
    )
    parser.add_argument(
        "--mode",
        choices=["nlte", "eos_idealgas", "all"],
        default="all",
        help="Generation mode: NLTE tables, ideal-gas EOS table, or all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parents[1] / out_dir

    if args.mode in ("nlte", "all"):
        if args.ntemp < 2 or args.ndens < 2 or args.ngroups < 1:
            raise ValueError("Require ntemp>=2, ndens>=2, ngroups>=1")
        if not (args.t_min > 0.0 and args.t_max > args.t_min):
            raise ValueError("Temperature range must satisfy 0 < t_min < t_max")
        if not (args.ni_min > 0.0 and args.ni_max > args.ni_min):
            raise ValueError("Density range must satisfy 0 < ni_min < ni_max")

        bounds_eV = np.asarray(args.bounds, dtype=np.float64)
        if bounds_eV.size != args.ngroups + 1:
            raise ValueError("bounds must contain ngroups+1 values")
        if not np.all(np.isfinite(bounds_eV)):
            raise ValueError("bounds must be finite")
        if np.any(np.diff(bounds_eV) <= 0.0):
            raise ValueError("bounds must be strictly increasing")

        temps_eV = np.logspace(np.log10(args.t_min), np.log10(args.t_max), args.ntemp)
        numdens_cm3 = np.logspace(np.log10(args.ni_min), np.log10(args.ni_max), args.ndens)

        # LTE: kappa_R = kappa_PA = kappa_PE = 100 cm^2/g
        kappa_lte = make_opacity_table(
            args.ntemp, args.ndens, args.ngroups, 100.0, temps_eV, numdens_cm3
        )
        lte_path = out_dir / "ionmix_lte_const.cn4"
        write_ionmix_cn4(
            output_path=lte_path,
            temps_eV=temps_eV,
            numdens_cm3=numdens_cm3,
            bounds_eV=bounds_eV,
            kappa_R=kappa_lte,
            kappa_PA=kappa_lte,
            kappa_PE=kappa_lte,
        )

        # Non-LTE simple: kappa_R = 50, kappa_PA = 100, kappa_PE = 200 cm^2/g
        kappa_r_nlte = make_opacity_table(
            args.ntemp, args.ndens, args.ngroups, 50.0, temps_eV, numdens_cm3
        )
        kappa_pa_nlte = make_opacity_table(
            args.ntemp, args.ndens, args.ngroups, 100.0, temps_eV, numdens_cm3
        )
        kappa_pe_nlte = make_opacity_table(
            args.ntemp, args.ndens, args.ngroups, 200.0, temps_eV, numdens_cm3
        )
        nlte_path = out_dir / "ionmix_nlte_simple.cn4"
        write_ionmix_cn4(
            output_path=nlte_path,
            temps_eV=temps_eV,
            numdens_cm3=numdens_cm3,
            bounds_eV=bounds_eV,
            kappa_R=kappa_r_nlte,
            kappa_PA=kappa_pa_nlte,
            kappa_PE=kappa_pe_nlte,
        )
        print(f"Generated: {lte_path}")
        print(f"Generated: {nlte_path}")

    if args.mode in ("eos_idealgas", "all"):
        eos_path = generate_eos_idealgas(out_dir)
        print(f"Generated: {eos_path}")


if __name__ == "__main__":
    main()
