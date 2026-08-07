#!/usr/bin/env python3
"""Convert IONMIX4/6 .cn4 binary files to TENRYU TMAT-H5."""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover - dependency guard
    h5py = None  # type: ignore[assignment]


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_tmat


PROTON_MASS_G = 1.67262192369e-24
J_TO_ERG = 1.0e7
MAX_RECORD_BYTES = 1 << 30


def _looks_like_record_length(length: int) -> bool:
    return 0 <= length <= MAX_RECORD_BYTES


def _ensure_h5py() -> None:
    if h5py is None:
        raise RuntimeError(
            "h5py is required for TMAT conversion. Install dependency: pip install h5py"
        )


@dataclass
class IonmixData:
    ntemp: int
    ndens: int
    ngroups: int
    z_header: np.ndarray
    frac_header: np.ndarray
    temps_eV: np.ndarray
    numdens_cm3: np.ndarray
    zbar: np.ndarray
    p_i_j_per_cm3: np.ndarray
    p_e_j_per_cm3: np.ndarray
    e_i_j_per_g: np.ndarray
    e_e_j_per_g: np.ndarray
    bounds_eV: np.ndarray
    kappa_R: np.ndarray
    kappa_PA: np.ndarray
    kappa_PE: np.ndarray
    file_endian: str


class FortranSequentialReader:
    """Reader for Fortran unformatted sequential records with endian autodetection."""

    def __init__(self, path: Path):
        self.path = path
        self._fp: BinaryIO = path.open("rb")
        self._file_size = path.stat().st_size
        self._file_endian: str | None = None
        self._host_endian = "little" if sys.byteorder == "little" else "big"

    @property
    def file_endian(self) -> str:
        if self._file_endian is None:
            return self._host_endian
        return self._file_endian

    def close(self) -> None:
        self._fp.close()

    def __enter__(self) -> "FortranSequentialReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def has_more_records(self) -> bool:
        pos = self._fp.tell()
        b = self._fp.read(1)
        self._fp.seek(pos)
        return b != b""

    def _read_exact(self, nbytes: int, label: str) -> bytes:
        data = self._fp.read(nbytes)
        if len(data) != nbytes:
            raise ValueError(
                f"Unexpected EOF while reading '{label}' in {self.path} at byte offset {self._fp.tell()}"
            )
        return data

    def _decode_marker(self, raw: bytes) -> int:
        if self._file_endian is None:
            swapped_endian = "big" if self._host_endian == "little" else "little"
            native = int.from_bytes(raw, byteorder=self._host_endian, signed=True)
            swapped = int.from_bytes(raw, byteorder=swapped_endian, signed=True)

            native_ok = self._probe_marker_candidate(native, self._host_endian)
            swapped_ok = self._probe_marker_candidate(swapped, swapped_endian)

            if native_ok and not swapped_ok:
                self._file_endian = self._host_endian
                return native
            if swapped_ok and not native_ok:
                self._file_endian = swapped_endian
                return swapped
            if native_ok and swapped_ok:
                self._file_endian = self._host_endian
                return native

            if _looks_like_record_length(native):
                self._file_endian = self._host_endian
                return native
            if _looks_like_record_length(swapped):
                self._file_endian = swapped_endian
                return swapped
            self._file_endian = self._host_endian
            return native
        return int.from_bytes(raw, byteorder=self._file_endian, signed=True)

    def _probe_marker_candidate(self, length: int, endian: str) -> bool:
        if not _looks_like_record_length(length):
            return False
        payload_start = self._fp.tell()
        trailer_offset = payload_start + length
        if trailer_offset + 4 > self._file_size:
            return False
        pos = self._fp.tell()
        try:
            self._fp.seek(trailer_offset)
            trailer = self._fp.read(4)
            if len(trailer) != 4:
                return False
            trailer_len = int.from_bytes(trailer, byteorder=endian, signed=True)
            return trailer_len == length
        finally:
            self._fp.seek(pos)

    def read_record_bytes(self, label: str) -> bytes:
        begin_raw = self._read_exact(4, f"{label} record marker")
        nbytes = self._decode_marker(begin_raw)
        if nbytes < 0:
            raise ValueError(f"Negative record length {nbytes} while reading '{label}'")
        if nbytes > MAX_RECORD_BYTES:
            raise ValueError(f"Record '{label}' is too large ({nbytes} bytes)")

        payload = self._read_exact(nbytes, f"{label} payload") if nbytes else b""
        end_raw = self._read_exact(4, f"{label} trailing marker")
        end_nbytes = self._decode_marker(end_raw)
        if end_nbytes != nbytes:
            raise ValueError(
                f"IONMIX record marker mismatch for '{label}': begin={nbytes}, end={end_nbytes}"
            )
        return payload

    def read_record_f64(self, label: str) -> np.ndarray:
        payload = self.read_record_bytes(label)
        if len(payload) % 8 != 0:
            raise ValueError(f"Record '{label}' has invalid byte size {len(payload)} for float64")
        if len(payload) == 0:
            return np.empty(0, dtype=np.float64)
        dtype = np.dtype("<f8" if self.file_endian == "little" else ">f8")
        return np.frombuffer(payload, dtype=dtype).astype(np.float64, copy=True)

    def read_int_from_f64_record(self, label: str) -> int:
        values = self.read_record_f64(label)
        if values.size != 1:
            raise ValueError(f"Record '{label}' must contain one float64 scalar, got {values.size}")
        value = float(values[0])
        if not np.isfinite(value):
            raise ValueError(f"Record '{label}' is non-finite")
        rounded = round(value)
        tol = 1.0e-10 * max(1.0, abs(value))
        if abs(value - rounded) > tol:
            raise ValueError(f"Record '{label}' is not integer-like: {value}")
        if rounded < 0:
            raise ValueError(f"Record '{label}' must be non-negative: {rounded}")
        return int(rounded)


def _check_axis_positive_increasing(name: str, values: np.ndarray) -> None:
    if values.ndim != 1:
        raise ValueError(f"{name} must be rank-1")
    if values.size < 1:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(values <= 0.0):
        raise ValueError(f"{name} values must be > 0")
    if values.size > 1 and np.any(np.diff(values) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")


def _check_bounds(bounds_eV: np.ndarray) -> None:
    if bounds_eV.ndim != 1 or bounds_eV.size < 2:
        raise ValueError("group bounds must be rank-1 with length >= 2")
    if not np.all(np.isfinite(bounds_eV)):
        raise ValueError("group bounds contain non-finite values")
    if bounds_eV[0] < 0.0:
        raise ValueError("group bounds first value must be >= 0")
    if np.any(bounds_eV[1:] <= 0.0):
        raise ValueError("group bounds after first must be > 0")
    if np.any(np.diff(bounds_eV) <= 0.0):
        raise ValueError("group bounds must be strictly increasing")


def _is_bounds_candidate(values: np.ndarray) -> bool:
    if values.ndim != 1 or values.size < 2:
        return False
    if not np.all(np.isfinite(values)):
        return False
    if values[0] < 0.0:
        return False
    if np.any(values[1:] <= 0.0):
        return False
    return bool(np.all(np.diff(values) > 0.0))


def _check_finite(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")


def _check_nonnegative(name: str, values: np.ndarray) -> None:
    _check_finite(name, values)
    if np.any(values < 0.0):
        raise ValueError(f"{name} contains negative values")


def read_ionmix_cn4(path: Path) -> IonmixData:
    with FortranSequentialReader(path) as reader:
        ntemp = reader.read_int_from_f64_record("ntemp")
        ndens = reader.read_int_from_f64_record("ndens")
        z_header = reader.read_record_f64("Z array")
        frac_header = reader.read_record_f64("fraction array")
        ngroups = reader.read_int_from_f64_record("ngroups")

        if ntemp <= 0 or ndens <= 0 or ngroups <= 0:
            raise ValueError(
                f"Invalid IONMIX header values: ntemp={ntemp}, ndens={ndens}, ngroups={ngroups}"
            )

        temps_eV = reader.read_record_f64("temperature grid")
        numdens_cm3 = reader.read_record_f64("ion number density grid")
        if temps_eV.size != ntemp:
            raise ValueError(f"temperature grid size mismatch: expected {ntemp}, got {temps_eV.size}")
        if numdens_cm3.size != ndens:
            raise ValueError(
                f"ion number density grid size mismatch: expected {ndens}, got {numdens_cm3.size}"
            )

        n2d = ndens * ntemp
        n3d = ngroups * ndens * ntemp
        eos_blocks: list[np.ndarray] = []
        for idx in range(12):
            block = reader.read_record_f64(f"EOS block {idx + 1}")
            if block.size != n2d:
                raise ValueError(
                    f"EOS block {idx + 1} size mismatch: expected {n2d}, got {block.size}"
                )
            eos_blocks.append(block)

        candidate = reader.read_record_f64("group bounds or optional entropy")
        expected_bounds = ngroups + 1
        if candidate.size == expected_bounds and _is_bounds_candidate(candidate):
            bounds_eV = candidate
        elif candidate.size == n2d:
            bounds_eV = reader.read_record_f64("group bounds")
        elif candidate.size == expected_bounds and n2d == expected_bounds:
            bounds_eV = reader.read_record_f64("group bounds")
        else:
            raise ValueError(
                "unable to parse group bounds record: expected ngroups+1 values "
                f"(={expected_bounds}) or optional entropy block of size {n2d}, got {candidate.size}"
            )
        if bounds_eV.size != expected_bounds:
            raise ValueError(
                f"group bounds size mismatch: expected {expected_bounds}, got {bounds_eV.size}"
            )

        kappa_r_flat = reader.read_record_f64("kappa_R")
        kappa_pa_flat = reader.read_record_f64("kappa_PA")
        if not reader.has_more_records():
            raise ValueError("missing Planck emission opacity table (kappa_PE)")
        kappa_pe_flat = reader.read_record_f64("kappa_PE")
        if reader.has_more_records():
            raise ValueError("unexpected trailing records after kappa_PE")

        if kappa_r_flat.size != n3d:
            raise ValueError(f"kappa_R size mismatch: expected {n3d}, got {kappa_r_flat.size}")
        if kappa_pa_flat.size != n3d:
            raise ValueError(f"kappa_PA size mismatch: expected {n3d}, got {kappa_pa_flat.size}")
        if kappa_pe_flat.size != n3d:
            raise ValueError(f"kappa_PE size mismatch: expected {n3d}, got {kappa_pe_flat.size}")

        zbar = eos_blocks[0].reshape((ndens, ntemp), order="C")
        p_i_j_per_cm3 = eos_blocks[2].reshape((ndens, ntemp), order="C")
        p_e_j_per_cm3 = eos_blocks[3].reshape((ndens, ntemp), order="C")
        e_i_j_per_g = eos_blocks[6].reshape((ndens, ntemp), order="C")
        e_e_j_per_g = eos_blocks[7].reshape((ndens, ntemp), order="C")

        kappa_r = kappa_r_flat.reshape((ngroups, ndens, ntemp), order="C")
        kappa_pa = kappa_pa_flat.reshape((ngroups, ndens, ntemp), order="C")
        kappa_pe = kappa_pe_flat.reshape((ngroups, ndens, ntemp), order="C")

        _check_axis_positive_increasing("temperature grid", temps_eV)
        _check_axis_positive_increasing("ion number density grid", numdens_cm3)
        _check_bounds(bounds_eV)
        _check_nonnegative("zbar", zbar)
        _check_finite("P_i", p_i_j_per_cm3)
        _check_finite("P_e", p_e_j_per_cm3)
        _check_finite("e_i", e_i_j_per_g)
        _check_finite("e_e", e_e_j_per_g)
        _check_nonnegative("kappa_R", kappa_r)
        _check_nonnegative("kappa_PA", kappa_pa)
        _check_nonnegative("kappa_PE", kappa_pe)

        return IonmixData(
            ntemp=ntemp,
            ndens=ndens,
            ngroups=ngroups,
            z_header=z_header,
            frac_header=frac_header,
            temps_eV=temps_eV,
            numdens_cm3=numdens_cm3,
            zbar=zbar,
            p_i_j_per_cm3=p_i_j_per_cm3,
            p_e_j_per_cm3=p_e_j_per_cm3,
            e_i_j_per_g=e_i_j_per_g,
            e_e_j_per_g=e_e_j_per_g,
            bounds_eV=bounds_eV,
            kappa_R=kappa_r,
            kappa_PA=kappa_pa,
            kappa_PE=kappa_pe,
            file_endian=reader.file_endian,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _create_float_dataset(group: h5py.Group, name: str, data: np.ndarray, units: str) -> None:
    arr = np.asarray(data, dtype=np.float64)
    kwargs: dict[str, object] = {}
    if arr.ndim > 0:
        kwargs["compression"] = "gzip"
        kwargs["compression_opts"] = 6
        kwargs["chunks"] = True
    ds = group.create_dataset(name, data=arr, **kwargs)
    ds.attrs["units"] = units


def _create_int32_dataset(group: h5py.Group, name: str, data: np.ndarray) -> None:
    group.create_dataset(name, data=np.asarray(data, dtype=np.int32))


def _create_string_scalar(group: h5py.Group, name: str, value: str) -> None:
    dt = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=value, dtype=dt)


def _create_string_vector(group: h5py.Group, name: str, values: list[str]) -> None:
    dt = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=dt)


def _detect_lte(kappa_pa: np.ndarray, kappa_pe: np.ndarray) -> tuple[bool, float]:
    denom = np.maximum(np.maximum(kappa_pa, kappa_pe), 1.0e-30)
    rel = np.abs(kappa_pa - kappa_pe) / denom
    max_rel = float(np.max(rel))
    return max_rel <= 1.0e-6, max_rel


def write_tmat_h5(
    output_path: Path,
    input_path: Path,
    material_name: str,
    atomic_number: int,
    mass_number: float,
    ionmix: IonmixData,
) -> tuple[bool, float]:
    _ensure_h5py()
    p_i_cgs = ionmix.p_i_j_per_cm3 * J_TO_ERG
    p_e_cgs = ionmix.p_e_j_per_cm3 * J_TO_ERG
    e_i_cgs = ionmix.e_i_j_per_g * J_TO_ERG
    e_e_cgs = ionmix.e_e_j_per_g * J_TO_ERG
    is_lte, max_rel = _detect_lte(ionmix.kappa_PA, ionmix.kappa_PE)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        h5.attrs["format_id"] = "tenryu.material_table.hdf5"
        h5.attrs["schema_version"] = "1.0.0"
        h5.attrs["units_system"] = "cgs_eV"
        h5.attrs["default_interpolation"] = "bilinear_loglog_clamp"

        material = h5.create_group("material")
        _create_string_scalar(material, "name", material_name)
        _create_int32_dataset(material, "Z", np.array([atomic_number], dtype=np.int32))
        _create_float_dataset(material, "A_amu", np.array([mass_number], dtype=np.float64), "amu")
        _create_float_dataset(
            material, "mass_fraction", np.array([1.0], dtype=np.float64), "dimensionless"
        )
        _create_float_dataset(
            material, "Abar_ion_amu", np.array(mass_number, dtype=np.float64), "amu"
        )
        _create_float_dataset(
            material, "number_fraction", np.array([1.0], dtype=np.float64), "dimensionless"
        )
        _create_string_vector(material, "species_name", [material_name])

        eos = h5.create_group("eos")
        eos.attrs["axis_order"] = "D,T"
        eos.attrs["primary_density_axis"] = "ni_cm3"
        eos_grid = eos.create_group("grid")
        _create_float_dataset(eos_grid, "ni_cm3", ionmix.numdens_cm3, "cm^-3")
        _create_float_dataset(eos_grid, "temperature_eV", ionmix.temps_eV, "eV")
        eos_fields = eos.create_group("fields")
        _create_float_dataset(eos_fields, "zbar", ionmix.zbar, "dimensionless")
        _create_float_dataset(eos_fields, "P_i", p_i_cgs, "dyne/cm^2")
        _create_float_dataset(eos_fields, "P_e", p_e_cgs, "dyne/cm^2")
        _create_float_dataset(eos_fields, "e_i", e_i_cgs, "erg/g")
        _create_float_dataset(eos_fields, "e_e", e_e_cgs, "erg/g")

        opacity = h5.create_group("opacity")
        opacity.attrs["axis_order"] = "G,D,T"
        opacity.attrs["primary_density_axis"] = "ni_cm3"
        opacity.attrs["is_lte"] = np.int32(1 if is_lte else 0)
        op_grid = opacity.create_group("grid")
        _create_float_dataset(op_grid, "ni_cm3", ionmix.numdens_cm3, "cm^-3")
        _create_float_dataset(op_grid, "temperature_eV", ionmix.temps_eV, "eV")
        _create_float_dataset(op_grid, "group_bounds_eV", ionmix.bounds_eV, "eV")
        op_fields = opacity.create_group("fields")
        _create_float_dataset(op_fields, "kappa_R", ionmix.kappa_R, "cm^2/g")
        _create_float_dataset(op_fields, "kappa_PA", ionmix.kappa_PA, "cm^2/g")
        _create_float_dataset(op_fields, "kappa_PE", ionmix.kappa_PE, "cm^2/g")

        provenance = h5.create_group("provenance")
        _create_string_scalar(provenance, "source_format", "IONMIX4/6")
        _create_string_vector(provenance, "source_files", [str(input_path)])
        _create_string_vector(provenance, "source_sha256", [_sha256(input_path)])
        _create_string_scalar(provenance, "generator_name", "ionmix_to_tmat.py")
        _create_string_scalar(provenance, "generator_version", "1.0.0")
        _create_string_scalar(provenance, "command_line", " ".join(sys.argv))
        _create_string_scalar(
            provenance,
            "notes",
            (
                "Converted from IONMIX with explicit CLI A/Z override. "
                f"Detected source endianness: {ionmix.file_endian}."
            ),
        )

    violations = validate_tmat.validate(str(output_path), strict=False)
    if violations:
        joined = "\n".join(violations)
        raise RuntimeError(f"TMAT validation failed after write:\n{joined}")
    return is_lte, max_rel


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert IONMIX4/6 .cn4 (Fortran binary) to TMAT-H5."
    )
    parser.add_argument("input_cn4", help="Path to input IONMIX .cn4 file")
    parser.add_argument("-o", "--output", required=True, help="Path to output .tmat.h5 file")
    parser.add_argument("--A", type=float, required=True, help="Mass number A [amu] (required)")
    parser.add_argument("--Z", type=int, required=True, help="Atomic number Z (required)")
    parser.add_argument(
        "--name",
        default=None,
        help="Material name (default: input file stem)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        input_path = Path(args.input_cn4)
        output_path = Path(args.output)
        if not input_path.exists():
            raise FileNotFoundError(f"input file not found: {input_path}")
        if args.A <= 0.0:
            raise ValueError("--A must be > 0")
        if args.Z <= 0:
            raise ValueError("--Z must be > 0")

        material_name = args.name if args.name else input_path.stem
        ionmix = read_ionmix_cn4(input_path)
        is_lte, max_rel = write_tmat_h5(
            output_path=output_path,
            input_path=input_path,
            material_name=material_name,
            atomic_number=int(args.Z),
            mass_number=float(args.A),
            ionmix=ionmix,
        )
        print(
            f"Wrote TMAT-H5: {output_path} "
            f"(is_lte={1 if is_lte else 0}, max_rel_PA_PE={max_rel:.3e})"
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
