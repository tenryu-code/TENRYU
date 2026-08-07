#!/usr/bin/env python3
"""Convert PROPACEOS .prp ASCII files to TENRYU TMAT-H5."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover - dependency guard
    h5py = None  # type: ignore[assignment]


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_tmat


J_TO_ERG = 1.0e7
EV_TO_ERG = 1.602176634e-12
FLOAT_TOKEN_RE = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][+-]?\d+)?")
INT_TOKEN_RE = re.compile(r"[+-]?\d+")


def _ensure_h5py() -> None:
    if h5py is None:
        raise RuntimeError(
            "h5py is required for TMAT conversion. Install dependency: pip install h5py"
        )


@dataclass
class PropaceosData:
    z_species: np.ndarray
    a_amu: np.ndarray
    number_fraction: np.ndarray
    mass_fraction: np.ndarray
    abar_ion_amu: float
    eos_t_eV: np.ndarray
    eos_ni_cm3: np.ndarray
    op_t_eV: np.ndarray
    op_ni_cm3: np.ndarray
    bounds_eV: np.ndarray
    zbar: np.ndarray
    p_i_dyne_cm2: np.ndarray
    p_e_dyne_cm2: np.ndarray
    e_i_j_g: np.ndarray
    e_e_j_g: np.ndarray
    kappa_R: np.ndarray
    kappa_PA: np.ndarray
    kappa_PE: np.ndarray
    a_source: str
    trailing_value_count: int
    ion_fractions: np.ndarray | None = None
    ion_stage_element: np.ndarray | None = None
    ion_stage_charge: np.ndarray | None = None


def _extract_floats(line: str) -> list[float]:
    values: list[float] = []
    for token in FLOAT_TOKEN_RE.findall(line):
        values.append(float(token.replace("D", "E").replace("d", "e")))
    return values


def _extract_ints(line: str) -> list[int]:
    return [int(token) for token in INT_TOKEN_RE.findall(line)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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
        raise ValueError("group_bounds_eV must be rank-1 with length >= 2")
    if not np.all(np.isfinite(bounds_eV)):
        raise ValueError("group_bounds_eV contains non-finite values")
    if bounds_eV[0] < 0.0:
        raise ValueError("first group boundary must be >= 0")
    if np.any(bounds_eV[1:] <= 0.0):
        raise ValueError("group boundaries after first must be > 0")
    if np.any(np.diff(bounds_eV) <= 0.0):
        raise ValueError("group_bounds_eV must be strictly increasing")


def _check_finite(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")


def _check_nonnegative(name: str, values: np.ndarray) -> None:
    _check_finite(name, values)
    if np.any(values < 0.0):
        raise ValueError(f"{name} contains negative values")


class TokenStream:
    def __init__(self, tokens: list[float]):
        self.tokens = tokens
        self.pos = 0

    def remaining(self) -> int:
        return len(self.tokens) - self.pos

    def _require(self, count: int, label: str) -> None:
        if count < 0:
            raise ValueError(f"invalid negative token count for {label}: {count}")
        if self.pos + count > len(self.tokens):
            raise ValueError(
                f"unexpected EOF while reading {label}: need {count} values, "
                f"have {len(self.tokens) - self.pos}"
            )

    def read_array(self, count: int, label: str) -> np.ndarray:
        self._require(count, label)
        out = np.asarray(self.tokens[self.pos : self.pos + count], dtype=np.float64)
        self.pos += count
        return out

    def skip(self, count: int, label: str) -> None:
        self._require(count, label)
        self.pos += count

    def read_int(self, label: str) -> int:
        value = float(self.read_array(1, label)[0])
        if not np.isfinite(value):
            raise ValueError(f"{label} is non-finite")
        rounded = round(value)
        tol = 1.0e-10 * max(1.0, abs(value))
        if abs(value - rounded) > tol:
            raise ValueError(f"{label} is not integer-like: {value}")
        return int(rounded)


def _infer_a_amu(
    header_lines: list[str], n_species: int, atomic_numbers: np.ndarray
) -> tuple[np.ndarray, str]:
    keyword_candidates = (
        "atomic weight",
        "atomic weights",
        "atomic mass",
        "atomic masses",
        "atomic wt",
        "amu",
    )
    for line in header_lines:
        lower = line.lower()
        if not any(key in lower for key in keyword_candidates):
            continue
        vals = _extract_floats(line)
        if len(vals) == n_species:
            arr = np.asarray(vals, dtype=np.float64)
            if np.all(np.isfinite(arr)) and np.all(arr > 0.0):
                return arr, "header keyword line"

    # Legacy converter uses line 31 for Z and line 34 for fractions.
    # Try line 32 as a common location for atomic masses if present.
    if len(header_lines) >= 32:
        vals = _extract_floats(header_lines[31])
        if len(vals) == n_species:
            arr = np.asarray(vals, dtype=np.float64)
            if np.all(np.isfinite(arr)) and np.all(arr > 0.0):
                return arr, "header line 32"

    return atomic_numbers.astype(np.float64), "fallback A=Z"


def _parse_header(
    lines: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, int, int, int]:
    if len(lines) < 41:
        raise ValueError("PROPACEOS file must contain at least 41 header lines")

    format_id_match = re.search(
        r"FORMAT_ID\s*=\s*(\d+)", "\n".join(lines[:41]), flags=re.IGNORECASE
    )
    if format_id_match is None:
        raise ValueError("failed to parse FORMAT_ID from PROPACEOS header")
    format_id = int(format_id_match.group(1))
    if format_id != 8:
        raise ValueError(f"unsupported PROPACEOS FORMAT_ID={format_id}; expected 8")

    write_eos_data = None
    for line in lines[:41]:
        if "write eos data" in line.lower():
            ints = _extract_ints(line)
            if ints:
                write_eos_data = ints[-1]
            break
    if write_eos_data is None:
        raise ValueError("failed to parse 'write eos data' from PROPACEOS header")
    if write_eos_data != 2:
        raise ValueError(
            f"unsupported write eos data mode {write_eos_data}; only mode 2 is supported"
        )

    write_ionization_fractions = None
    write_populations = None
    for line in lines[:41]:
        lower = line.lower()
        if "write ionization fractions" in lower:
            ints = _extract_ints(line)
            if ints:
                write_ionization_fractions = ints[-1]
        elif "write populations" in lower:
            ints = _extract_ints(line)
            if ints:
                write_populations = ints[-1]
    if write_ionization_fractions is None:
        raise ValueError(
            "failed to parse 'write ionization fractions' from PROPACEOS header"
        )
    if write_populations is None:
        raise ValueError("failed to parse 'write populations' from PROPACEOS header")
    if write_populations != 0:
        raise ValueError("write populations != 0 is not supported")

    element_count = None
    line20_ints = _extract_ints(lines[19])
    if line20_ints:
        element_count = line20_ints[0]

    atomic_numbers = [value for value in _extract_ints(lines[30]) if value > 0]
    fractions = _extract_floats(lines[33])

    if element_count is None:
        element_count = len(atomic_numbers)
    if element_count <= 0:
        raise ValueError("failed to parse element count from PROPACEOS header")
    if len(atomic_numbers) != element_count:
        raise ValueError(
            "atomic number count mismatch in PROPACEOS header: "
            f"expected {element_count}, got {len(atomic_numbers)}"
        )
    if len(fractions) != element_count:
        raise ValueError(
            "fraction count mismatch in PROPACEOS header: "
            f"expected {element_count}, got {len(fractions)}"
        )

    z_species = np.asarray(atomic_numbers, dtype=np.int32)
    number_fraction = np.asarray(fractions, dtype=np.float64)
    if np.any(~np.isfinite(number_fraction)):
        raise ValueError("number fractions contain non-finite values")
    if np.any(number_fraction < 0.0):
        raise ValueError("number fractions contain negative values")
    total = float(np.sum(number_fraction))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("number fractions sum must be > 0")
    number_fraction /= total

    a_amu, a_source = _infer_a_amu(lines[:41], element_count, z_species)
    if np.any(~np.isfinite(a_amu)) or np.any(a_amu <= 0.0):
        raise ValueError("A_amu values must be finite and > 0")

    mass_fraction = number_fraction * a_amu
    mass_total = float(np.sum(mass_fraction))
    if not np.isfinite(mass_total) or mass_total <= 0.0:
        raise ValueError("derived mass fractions are invalid")
    mass_fraction /= mass_total

    return (
        z_species,
        a_amu,
        number_fraction,
        a_source,
        write_eos_data,
        write_ionization_fractions,
        write_populations,
    )


def _read_numeric_tokens(lines: list[str]) -> list[float]:
    tokens: list[float] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("*"):
            continue
        tokens.extend(_extract_floats(line))
    return tokens


def _find_data_start(lines: list[str]) -> int:
    """Return 0-based line index where the numeric data section begins.

    Scans for the star-comment block containing 'mesh parameters' that
    precedes the EOS grid data, then returns the index of the first
    non-comment line after it.
    """
    for i in range(34, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("*") and "mesh" in stripped.lower():
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("*"):
                j += 1
            return j
    return 41  # fallback for legacy files


def read_propaceos_prp(path: Path) -> PropaceosData:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    (
        z_species,
        a_amu,
        number_fraction,
        a_source,
        write_eos_data,
        write_ionization_fractions,
        _write_populations,
    ) = _parse_header(lines)
    data_start = _find_data_start(lines)
    tokens = _read_numeric_tokens(lines[data_start:])
    stream = TokenStream(tokens)

    ntemp_eos = stream.read_int("EOS temperature count")
    eos_t = stream.read_array(ntemp_eos, "EOS temperature grid")
    ndens_eos = stream.read_int("EOS density count")
    eos_rho = stream.read_array(ndens_eos, "EOS density grid")
    stream.skip(1, "rho0")

    ntemp_op = stream.read_int("opacity temperature count")
    op_t = stream.read_array(ntemp_op, "opacity temperature grid")
    ndens_op = stream.read_int("opacity density count")
    op_rho = stream.read_array(ndens_op, "opacity density grid")

    ngroups = stream.read_int("opacity group count")
    if ntemp_eos <= 0 or ndens_eos <= 0 or ntemp_op <= 0 or ndens_op <= 0 or ngroups <= 0:
        raise ValueError(
            "invalid grid counts parsed from PROPACEOS: "
            f"ntemp_eos={ntemp_eos}, ndens_eos={ndens_eos}, "
            f"ntemp_op={ntemp_op}, ndens_op={ndens_op}, ngroups={ngroups}"
        )

    bounds = stream.read_array(ngroups + 1, "group bounds")
    n2d_eos = ntemp_eos * ndens_eos
    n2d_op = ntemp_op * ndens_op

    if write_ionization_fractions == 1:
        ion_fraction_blocks: list[np.ndarray] = []
        ion_stage_elements: list[np.ndarray] = []
        ion_stage_charges: list[np.ndarray] = []
        max_ion_sum_error = 0.0
        for element_index, z_value in enumerate(z_species.tolist()):
            z = int(z_value)
            raw = stream.read_array(
                n2d_eos * (z + 1), f"ionization fractions Z={z}"
            )
            frac_e = raw.reshape(
                (ndens_eos, ntemp_eos, z + 1), order="C"
            ).transpose(2, 0, 1)
            _check_finite(f"ionization fractions Z={z}", frac_e)
            if np.any(frac_e < -1.0e-12):
                raise ValueError(
                    f"ionization fractions Z={z} contain values below -1e-12 "
                    f"(min={float(np.min(frac_e)):.3e})"
                )
            np.maximum(frac_e, 0.0, out=frac_e)

            stage_sum = np.sum(frac_e, axis=0)
            sum_error = np.abs(stage_sum - 1.0)
            max_ion_sum_error = max(max_ion_sum_error, float(np.max(sum_error)))
            bad_count = int(np.count_nonzero(sum_error > 0.2))
            if bad_count > 0:
                raise ValueError(
                    f"ionization fractions element {element_index} (Z={z}) have "
                    f"stage sums outside |S-1| <= 0.2 at {bad_count} grid points"
                )

            ion_fraction_blocks.append(frac_e)
            ion_stage_elements.append(
                np.full(z + 1, element_index, dtype=np.int32)
            )
            ion_stage_charges.append(np.arange(z + 1, dtype=np.int32))

        ion_fractions = np.concatenate(ion_fraction_blocks, axis=0)
        ion_stage_element = np.concatenate(ion_stage_elements)
        ion_stage_charge = np.concatenate(ion_stage_charges)
        if max_ion_sum_error > 1.0e-3:
            print(
                "WARNING: ionization fraction stage sums have "
                f"max |S-1|={max_ion_sum_error:.3e}",
                file=sys.stderr,
            )
    else:
        ion_fractions = None
        ion_stage_element = None
        ion_stage_charge = None

    zbar_flat = stream.read_array(n2d_eos, "zbar")
    stream.skip(3 * n2d_op, "integrated opacity blocks")
    if write_eos_data == 2:
        e_int = stream.read_array(n2d_eos, "total internal energy")
        e_i = stream.read_array(n2d_eos, "ion energy")
        e_e = stream.read_array(n2d_eos, "electron energy")
        p_i = stream.read_array(n2d_eos, "ion pressure")
        p_e = stream.read_array(n2d_eos, "electron pressure")
    else:
        raise ValueError(f"unsupported write eos data mode {write_eos_data}")

    rosseland = np.empty((n2d_op, ngroups), dtype=np.float64)
    emission = np.empty((n2d_op, ngroups), dtype=np.float64)
    absorption = np.empty((n2d_op, ngroups), dtype=np.float64)
    for td_idx in range(n2d_op):
        rosseland[td_idx, :] = stream.read_array(ngroups, f"kappa_R row {td_idx}")
        emission[td_idx, :] = stream.read_array(ngroups, f"kappa_PE row {td_idx}")
        absorption[td_idx, :] = stream.read_array(ngroups, f"kappa_PA row {td_idx}")

    trailing = stream.remaining()
    if trailing > 0:
        raise ValueError(
            f"PROPACEOS file has {trailing} unread trailing values; "
            "this may indicate a format parsing error"
        )

    zbar = zbar_flat.reshape((ndens_eos, ntemp_eos), order="C")
    p_i_2d = p_i.reshape((ndens_eos, ntemp_eos), order="C")
    p_e_2d = p_e.reshape((ndens_eos, ntemp_eos), order="C")
    e_int_2d = e_int.reshape((ndens_eos, ntemp_eos), order="C")
    e_i_2d = e_i.reshape((ndens_eos, ntemp_eos), order="C")
    e_e_2d = e_e.reshape((ndens_eos, ntemp_eos), order="C")
    e_sum = e_i_2d + e_e_2d
    max_rel_err = float(
        np.max(np.abs(e_int_2d - e_sum) / np.maximum(np.abs(e_int_2d), 1.0e-30))
    )
    if max_rel_err > 1.0e-6:
        raise ValueError(
            f"Eint != Eion + Eele consistency check failed (max_rel={max_rel_err:.3e})"
        )

    kappa_r = np.empty((ngroups, ndens_op, ntemp_op), dtype=np.float64)
    kappa_pa = np.empty((ngroups, ndens_op, ntemp_op), dtype=np.float64)
    kappa_pe = np.empty((ngroups, ndens_op, ntemp_op), dtype=np.float64)
    for g in range(ngroups):
        kappa_r[g, :, :] = rosseland[:, g].reshape((ndens_op, ntemp_op), order="C")
        kappa_pa[g, :, :] = absorption[:, g].reshape((ndens_op, ntemp_op), order="C")
        kappa_pe[g, :, :] = emission[:, g].reshape((ndens_op, ntemp_op), order="C")

    _check_axis_positive_increasing("EOS temperature grid", eos_t)
    _check_axis_positive_increasing("EOS density grid", eos_rho)
    _check_axis_positive_increasing("opacity temperature grid", op_t)
    _check_axis_positive_increasing("opacity density grid", op_rho)
    _check_bounds(bounds)
    _check_nonnegative("zbar", zbar)
    _check_finite("P_i", p_i_2d)
    _check_finite("P_e", p_e_2d)
    _check_finite("e_i", e_i_2d)
    _check_finite("e_e", e_e_2d)
    for name, arr in [("kappa_R", kappa_r), ("kappa_PA", kappa_pa), ("kappa_PE", kappa_pe)]:
        _check_finite(name, arr)
        neg_count = int(np.sum(arr < 0.0))
        if neg_count > 0:
            print(
                f"WARNING: {name} has {neg_count} negative values "
                f"(min={float(np.min(arr)):.3e}); clamping to zero",
                file=sys.stderr,
            )
            np.maximum(arr, 0.0, out=arr)

    abar = float(np.sum(number_fraction * a_amu))
    if not np.isfinite(abar) or abar <= 0.0:
        raise ValueError("Abar_ion_amu is invalid")

    mass_fraction = number_fraction * a_amu
    mass_fraction /= np.sum(mass_fraction)

    return PropaceosData(
        z_species=z_species,
        a_amu=a_amu,
        number_fraction=number_fraction,
        mass_fraction=mass_fraction,
        abar_ion_amu=abar,
        eos_t_eV=eos_t,
        eos_ni_cm3=eos_rho,
        op_t_eV=op_t,
        op_ni_cm3=op_rho,
        bounds_eV=bounds,
        zbar=zbar,
        p_i_dyne_cm2=p_i_2d,
        p_e_dyne_cm2=p_e_2d,
        e_i_j_g=e_i_2d,
        e_e_j_g=e_e_2d,
        kappa_R=kappa_r,
        kappa_PA=kappa_pa,
        kappa_PE=kappa_pe,
        a_source=a_source,
        trailing_value_count=trailing,
        ion_fractions=ion_fractions,
        ion_stage_element=ion_stage_element,
        ion_stage_charge=ion_stage_charge,
    )


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
    data: PropaceosData,
    no_ionization: bool = False,
    kirchhoff_pe: bool = False,
) -> tuple[bool, float]:
    _ensure_h5py()
    e_i_cgs = data.e_i_j_g * J_TO_ERG
    e_e_cgs = data.e_e_j_g * J_TO_ERG
    p_i_cgs = data.p_i_dyne_cm2  # PROPACEOS pressure is already dyne/cm^2
    p_e_cgs = data.p_e_dyne_cm2
    is_lte, max_rel = _detect_lte(data.kappa_PA, data.kappa_PE)
    ionization_written = data.ion_fractions is not None and not no_ionization

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        h5.attrs["format_id"] = "tenryu.material_table.hdf5"
        h5.attrs["schema_version"] = "1.0.0"
        h5.attrs["units_system"] = "cgs_eV"
        h5.attrs["default_interpolation"] = "bilinear_loglog_clamp"

        material = h5.create_group("material")
        _create_string_scalar(material, "name", material_name)
        _create_int32_dataset(material, "Z", data.z_species)
        _create_float_dataset(material, "A_amu", data.a_amu, "amu")
        _create_float_dataset(material, "mass_fraction", data.mass_fraction, "dimensionless")
        _create_float_dataset(
            material,
            "Abar_ion_amu",
            np.array(data.abar_ion_amu, dtype=np.float64),
            "amu",
        )
        _create_float_dataset(
            material, "number_fraction", data.number_fraction, "dimensionless"
        )

        eos = h5.create_group("eos")
        eos.attrs["axis_order"] = "D,T"
        eos.attrs["primary_density_axis"] = "ni_cm3"
        eos_grid = eos.create_group("grid")
        _create_float_dataset(eos_grid, "ni_cm3", data.eos_ni_cm3, "cm^-3")
        _create_float_dataset(eos_grid, "temperature_eV", data.eos_t_eV, "eV")
        eos_fields = eos.create_group("fields")
        _create_float_dataset(eos_fields, "zbar", data.zbar, "dimensionless")
        _create_float_dataset(eos_fields, "P_i", p_i_cgs, "dyne/cm^2")
        _create_float_dataset(eos_fields, "P_e", p_e_cgs, "dyne/cm^2")
        _create_float_dataset(eos_fields, "e_i", e_i_cgs, "erg/g")
        _create_float_dataset(eos_fields, "e_e", e_e_cgs, "erg/g")

        if ionization_written:
            ionization = h5.create_group("ionization")
            ionization.attrs["axis_order"] = "S,D,T"
            ionization.attrs["primary_density_axis"] = "ni_cm3"
            ion_grid = ionization.create_group("grid")
            _create_float_dataset(ion_grid, "ni_cm3", data.eos_ni_cm3, "cm^-3")
            _create_float_dataset(
                ion_grid, "temperature_eV", data.eos_t_eV, "eV"
            )
            _create_int32_dataset(
                ionization, "stage_element", data.ion_stage_element
            )
            _create_int32_dataset(
                ionization, "stage_charge", data.ion_stage_charge
            )
            ion_fields = ionization.create_group("fields")
            _create_float_dataset(
                ion_fields, "fractions", data.ion_fractions, "fraction"
            )

        opacity = h5.create_group("opacity")
        opacity.attrs["axis_order"] = "G,D,T"
        opacity.attrs["primary_density_axis"] = "ni_cm3"
        opacity.attrs["is_lte"] = np.int32(1 if is_lte else 0)
        op_grid = opacity.create_group("grid")
        _create_float_dataset(op_grid, "ni_cm3", data.op_ni_cm3, "cm^-3")
        _create_float_dataset(op_grid, "temperature_eV", data.op_t_eV, "eV")
        _create_float_dataset(op_grid, "group_bounds_eV", data.bounds_eV, "eV")
        op_fields = opacity.create_group("fields")
        _create_float_dataset(op_fields, "kappa_R", data.kappa_R, "cm^2/g")
        _create_float_dataset(op_fields, "kappa_PA", data.kappa_PA, "cm^2/g")
        _create_float_dataset(op_fields, "kappa_PE", data.kappa_PE, "cm^2/g")

        provenance = h5.create_group("provenance")
        _create_string_scalar(provenance, "source_format", "PROPACEOS")
        _create_string_vector(provenance, "source_files", [str(input_path)])
        _create_string_vector(provenance, "source_sha256", [_sha256(input_path)])
        _create_string_scalar(provenance, "generator_name", "propaceos_to_tmat.py")
        _create_string_scalar(provenance, "generator_version", "1.0.0")
        _create_string_scalar(provenance, "command_line", " ".join(sys.argv))
        _create_string_scalar(
            provenance,
            "notes",
            (
                "PROPACEOS conversion: pressure kept in dyne/cm^2, "
                "energy converted J/g -> erg/g. "
                f"A_amu source={data.a_source}; "
                f"trailing_numeric_values={data.trailing_value_count}; "
                f"ionization_written={1 if ionization_written else 0}; "
                f"kirchhoff_pe={1 if kirchhoff_pe else 0}."
            ),
        )

    p_i_ideal_hot = float(data.eos_ni_cm3[0] * data.eos_t_eV[-1] * EV_TO_ERG)
    p_i_hot = float(p_i_cgs[0, -1])
    rel_err_hot = abs(p_i_hot - p_i_ideal_hot) / max(abs(p_i_ideal_hot), 1.0e-30)
    if rel_err_hot > 0.5:
        raise ValueError(
            "low-density ideal-gas sanity check failed for P_i: "
            f"rel_err={rel_err_hot:.3e} at ni={data.eos_ni_cm3[0]:.6e} cm^-3, "
            f"T={data.eos_t_eV[-1]:.6e} eV"
        )

    violations = validate_tmat.validate(str(output_path), strict=False)
    if violations:
        raise RuntimeError("TMAT validation failed after write:\n" + "\n".join(violations))
    return is_lte, max_rel


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert PROPACEOS .prp ASCII to TMAT-H5.")
    parser.add_argument("input_prp", help="Path to input .prp file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to output .tmat.h5 file (default: input path with .tmat.h5 extension)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Material name (default: input file stem)",
    )
    parser.add_argument(
        "--no-ionization",
        action="store_true",
        help="Omit the /ionization group even when ionization fractions are present.",
    )
    parser.add_argument(
        "--kirchhoff-pe",
        action="store_true",
        help="Set kappa_PE equal to kappa_PA before writing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        input_path = Path(args.input_prp)
        output_path = Path(args.output) if args.output else input_path.with_suffix(".tmat.h5")
        if not input_path.exists():
            raise FileNotFoundError(f"input file not found: {input_path}")

        material_name = args.name if args.name else input_path.stem
        parsed = read_propaceos_prp(input_path)
        if args.kirchhoff_pe:
            parsed.kappa_PE = parsed.kappa_PA.copy()
        is_lte, max_rel = write_tmat_h5(
            output_path,
            input_path,
            material_name,
            parsed,
            no_ionization=args.no_ionization,
            kirchhoff_pe=args.kirchhoff_pe,
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
