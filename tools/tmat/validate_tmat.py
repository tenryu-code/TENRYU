#!/usr/bin/env python3
"""Standalone validator for TENRYU TMAT-H5 files (spec v1.0)."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover - dependency guard
    h5py = None  # type: ignore[assignment]


FORMAT_ID = "tenryu.material_table.hdf5"
UNITS_SYSTEM = "cgs_eV"
SUPPORTED_SCHEMA_MAJOR = 1
SUPPORTED_REQUIRED_FEATURES: set[str] = set()
FLOAT64 = np.dtype(np.float64)
INT32 = np.dtype(np.int32)
SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
FEATURE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)_v([0-9]+)$")

# HDF5 built-in filters accepted by TENRYU validator:
# 1=deflate(gzip), 2=shuffle, 3=fletcher32
ALLOWED_FILTER_IDS = {1, 2, 3}
ALLOWED_COMPRESSION = {None, "gzip"}


@dataclass
class ValidationContext:
    strict: bool = False
    violations: list[str] = field(default_factory=list)

    def add(self, code: str, path: str, message: str) -> None:
        self.violations.append(f"{code} {path}: {message}")


def _ensure_h5py() -> None:
    if h5py is None:
        raise RuntimeError(
            "h5py is required for TMAT validation. Install dependency: pip install h5py"
        )


def _attr_path(group_path: str, attr: str) -> str:
    if group_path == "/":
        return f"/@{attr}"
    return f"{group_path}/@{attr}"


def _decode_utf8_scalar(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise TypeError("expected scalar value")
        return _decode_utf8_scalar(value.item())
    if isinstance(value, np.generic):
        return _decode_utf8_scalar(value.item())
    raise TypeError(f"expected UTF-8 string scalar, got {type(value).__name__}")


def _decode_utf8_vector(value: Any) -> list[str]:
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise TypeError("expected rank-1 string array")
    out: list[str] = []
    for item in arr.tolist():
        out.append(_decode_utf8_scalar(item))
    return out


def _read_attr_string(
    ctx: ValidationContext,
    obj: h5py.Group | h5py.Dataset,
    obj_path: str,
    name: str,
    required: bool = False,
    strict_required: bool = False,
) -> str | None:
    path = _attr_path(obj_path, name)
    if name not in obj.attrs:
        if required or strict_required:
            ctx.add("TMAT_E001", path, "required attribute is missing")
        return None
    try:
        return _decode_utf8_scalar(obj.attrs[name])
    except UnicodeDecodeError as exc:
        ctx.add("TMAT_E011", path, f"invalid UTF-8: {exc}")
    except TypeError as exc:
        ctx.add("TMAT_E005", path, str(exc))
    return None


def _read_attr_int32_scalar(
    ctx: ValidationContext,
    obj: h5py.Group | h5py.Dataset,
    obj_path: str,
    name: str,
    required: bool = False,
    strict_required: bool = False,
) -> int | None:
    path = _attr_path(obj_path, name)
    if name not in obj.attrs:
        if required or strict_required:
            ctx.add("TMAT_E001", path, "required attribute is missing")
        return None

    raw = np.asarray(obj.attrs[name])
    if raw.shape != ():
        ctx.add("TMAT_E006", path, f"expected scalar attribute, got shape {raw.shape}")
        return None
    if raw.dtype != INT32:
        ctx.add("TMAT_E005", path, f"expected int32 attribute, got {raw.dtype}")
        return None
    return int(raw.item())


def _get_group(
    ctx: ValidationContext,
    h5: h5py.File,
    path: str,
    required: bool = False,
    strict_required: bool = False,
) -> h5py.Group | None:
    obj = h5.get(path)
    if obj is None:
        if required or strict_required:
            ctx.add("TMAT_E001", path, "required group is missing")
        return None
    if not isinstance(obj, h5py.Group):
        ctx.add("TMAT_E005", path, f"expected HDF5 group, got {type(obj).__name__}")
        return None
    return obj


def _get_dataset(
    ctx: ValidationContext,
    h5: h5py.File,
    path: str,
    required: bool = False,
    strict_required: bool = False,
) -> h5py.Dataset | None:
    obj = h5.get(path)
    if obj is None:
        if required or strict_required:
            ctx.add("TMAT_E001", path, "required dataset is missing")
        return None
    if not isinstance(obj, h5py.Dataset):
        ctx.add("TMAT_E005", path, f"expected HDF5 dataset, got {type(obj).__name__}")
        return None
    return obj


def _read_float64_dataset(
    ctx: ValidationContext,
    h5: h5py.File,
    path: str,
    rank: int | None = None,
    shape: tuple[int, ...] | None = None,
    required: bool = False,
    strict_required: bool = False,
) -> np.ndarray | None:
    ds = _get_dataset(ctx, h5, path, required=required, strict_required=strict_required)
    if ds is None:
        return None
    if ds.dtype != FLOAT64:
        ctx.add("TMAT_E005", path, f"expected float64 dataset, got {ds.dtype}")
    if rank is not None and ds.ndim != rank:
        ctx.add("TMAT_E006", path, f"expected rank {rank}, got rank {ds.ndim}")
    if shape is not None and tuple(ds.shape) != tuple(shape):
        ctx.add("TMAT_E006", path, f"expected shape {shape}, got {tuple(ds.shape)}")
    try:
        return np.asarray(ds[...], dtype=np.float64)
    except Exception as exc:  # pragma: no cover - defensive
        ctx.add("TMAT_E005", path, f"failed to read dataset: {exc}")
        return None


def _read_int32_dataset(
    ctx: ValidationContext,
    h5: h5py.File,
    path: str,
    rank: int | None = None,
    shape: tuple[int, ...] | None = None,
    required: bool = False,
    strict_required: bool = False,
) -> np.ndarray | None:
    ds = _get_dataset(ctx, h5, path, required=required, strict_required=strict_required)
    if ds is None:
        return None
    if ds.dtype != INT32:
        ctx.add("TMAT_E005", path, f"expected int32 dataset, got {ds.dtype}")
    if rank is not None and ds.ndim != rank:
        ctx.add("TMAT_E006", path, f"expected rank {rank}, got rank {ds.ndim}")
    if shape is not None and tuple(ds.shape) != tuple(shape):
        ctx.add("TMAT_E006", path, f"expected shape {shape}, got {tuple(ds.shape)}")
    try:
        return np.asarray(ds[...], dtype=np.int64)
    except Exception as exc:  # pragma: no cover - defensive
        ctx.add("TMAT_E005", path, f"failed to read dataset: {exc}")
        return None


def _read_string_scalar_dataset(
    ctx: ValidationContext,
    h5: h5py.File,
    path: str,
    required: bool = False,
    strict_required: bool = False,
) -> str | None:
    ds = _get_dataset(ctx, h5, path, required=required, strict_required=strict_required)
    if ds is None:
        return None
    if h5py.check_string_dtype(ds.dtype) is None:
        ctx.add("TMAT_E005", path, f"expected UTF-8 string dataset, got {ds.dtype}")
        return None
    if ds.ndim != 0:
        ctx.add("TMAT_E006", path, f"expected scalar string dataset, got rank {ds.ndim}")
        return None
    try:
        return _decode_utf8_scalar(ds[()])
    except UnicodeDecodeError as exc:
        ctx.add("TMAT_E011", path, f"invalid UTF-8: {exc}")
    except TypeError as exc:
        ctx.add("TMAT_E005", path, str(exc))
    return None


def _read_string_vector_dataset(
    ctx: ValidationContext,
    h5: h5py.File,
    path: str,
    required: bool = False,
    strict_required: bool = False,
) -> list[str] | None:
    ds = _get_dataset(ctx, h5, path, required=required, strict_required=strict_required)
    if ds is None:
        return None
    if h5py.check_string_dtype(ds.dtype) is None:
        ctx.add("TMAT_E005", path, f"expected UTF-8 string dataset, got {ds.dtype}")
        return None
    if ds.ndim != 1:
        ctx.add("TMAT_E006", path, f"expected rank-1 string dataset, got rank {ds.ndim}")
        return None
    try:
        return _decode_utf8_vector(ds[...])
    except UnicodeDecodeError as exc:
        ctx.add("TMAT_E011", path, f"invalid UTF-8: {exc}")
    except TypeError as exc:
        ctx.add("TMAT_E006", path, str(exc))
    return None


def _check_finite(ctx: ValidationContext, path: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        ctx.add("TMAT_E007", path, "contains non-finite values")


def _check_positive_axis(ctx: ValidationContext, path: str, values: np.ndarray) -> None:
    _check_finite(ctx, path, values)
    if np.any(values <= 0.0):
        ctx.add("TMAT_E009", path, "axis must be > 0")
    if values.size >= 2 and np.any(np.diff(values) <= 0.0):
        ctx.add("TMAT_E008", path, "axis must be strictly increasing")


def _check_group_bounds_axis(ctx: ValidationContext, path: str, values: np.ndarray) -> None:
    _check_finite(ctx, path, values)
    if values.size < 2:
        ctx.add("TMAT_E006", path, "group_bounds_eV must have length nG+1 with nG>=1")
        return
    if values[0] < 0.0:
        ctx.add("TMAT_E009", path, "first group bound must be >= 0")
    if np.any(values[1:] <= 0.0):
        ctx.add("TMAT_E009", path, "group bounds after first must be > 0")
    if np.any(np.diff(values) <= 0.0):
        ctx.add("TMAT_E008", path, "group bounds must be strictly increasing")


def _validate_root(ctx: ValidationContext, h5: h5py.File) -> list[str]:
    required_features: list[str] = []

    format_id = _read_attr_string(ctx, h5, "/", "format_id", required=True)
    if format_id is not None and format_id != FORMAT_ID:
        ctx.add("TMAT_E001", "/@format_id", f"expected '{FORMAT_ID}', got '{format_id}'")

    schema = _read_attr_string(ctx, h5, "/", "schema_version", required=True)
    if schema is not None:
        match = SEMVER_RE.match(schema)
        if not match:
            ctx.add("TMAT_E002", "/@schema_version", f"invalid semantic version '{schema}'")
        else:
            major = int(match.group(1))
            if major != SUPPORTED_SCHEMA_MAJOR:
                ctx.add(
                    "TMAT_E002",
                    "/@schema_version",
                    f"unsupported major version {major} (expected {SUPPORTED_SCHEMA_MAJOR})",
                )

    units = _read_attr_string(ctx, h5, "/", "units_system", required=True)
    if units is not None and units != UNITS_SYSTEM:
        ctx.add("TMAT_E003", "/@units_system", f"expected '{UNITS_SYSTEM}', got '{units}'")

    required_features_raw = _read_attr_string(
        ctx, h5, "/", "required_features", required=False, strict_required=False
    )
    if required_features_raw:
        tokens = [token.strip() for token in required_features_raw.split(",") if token.strip()]
        for token in tokens:
            if FEATURE_RE.match(token) is None:
                ctx.add("TMAT_E004", "/@required_features", f"malformed token '{token}'")
                continue
            required_features.append(token)
            if token not in SUPPORTED_REQUIRED_FEATURES:
                ctx.add("TMAT_E004", "/@required_features", f"unknown required feature '{token}'")

    _read_attr_string(
        ctx,
        h5,
        "/",
        "default_interpolation",
        required=False,
        strict_required=ctx.strict,
    )
    return required_features


def _validate_material(ctx: ValidationContext, h5: h5py.File) -> int | None:
    material = _get_group(ctx, h5, "/material", required=True)
    if material is None:
        return None

    name = _read_string_scalar_dataset(
        ctx, h5, "/material/name", required=False, strict_required=ctx.strict
    )
    if name is not None and name.strip() == "":
        ctx.add("TMAT_E006", "/material/name", "material name should be non-empty")

    z = _read_int32_dataset(ctx, h5, "/material/Z", rank=1, required=True)
    a_amu = _read_float64_dataset(ctx, h5, "/material/A_amu", rank=1, required=True)
    mass_fraction = _read_float64_dataset(
        ctx, h5, "/material/mass_fraction", rank=1, required=True
    )
    abar = _read_float64_dataset(ctx, h5, "/material/Abar_ion_amu", rank=0, required=True)

    n_species: int | None = None
    if z is not None:
        n_species = int(z.size)
        if n_species < 1:
            ctx.add("TMAT_E009", "/material/Z", "n_species must be >= 1")
        if np.any(z < 1):
            ctx.add("TMAT_E009", "/material/Z", "all Z values must be >= 1")

    if a_amu is not None:
        _check_finite(ctx, "/material/A_amu", a_amu)
        if np.any(a_amu <= 0.0):
            ctx.add("TMAT_E009", "/material/A_amu", "all A_amu values must be > 0")

    if mass_fraction is not None:
        _check_finite(ctx, "/material/mass_fraction", mass_fraction)
        if np.any(mass_fraction < 0.0):
            ctx.add("TMAT_E009", "/material/mass_fraction", "mass_fraction must be >= 0")
        mass_sum = float(np.sum(mass_fraction))
        if not np.isfinite(mass_sum) or abs(mass_sum - 1.0) > 1.0e-8:
            ctx.add(
                "TMAT_E010",
                "/material/mass_fraction",
                f"sum must be 1 within 1e-8, got {mass_sum:.16e}",
            )

    if abar is not None:
        val = float(abar.item())
        if not np.isfinite(val):
            ctx.add("TMAT_E007", "/material/Abar_ion_amu", "must be finite")
        elif val <= 0.0:
            ctx.add("TMAT_E009", "/material/Abar_ion_amu", "must be > 0")

    if n_species is not None:
        if a_amu is not None and a_amu.shape != (n_species,):
            ctx.add(
                "TMAT_E006",
                "/material/A_amu",
                f"shape mismatch with /material/Z: expected ({n_species},), got {a_amu.shape}",
            )
        if mass_fraction is not None and mass_fraction.shape != (n_species,):
            ctx.add(
                "TMAT_E006",
                "/material/mass_fraction",
                f"shape mismatch with /material/Z: expected ({n_species},), got {mass_fraction.shape}",
            )

    number_fraction = _read_float64_dataset(
        ctx,
        h5,
        "/material/number_fraction",
        rank=1,
        required=False,
        strict_required=ctx.strict,
    )
    if number_fraction is not None:
        _check_finite(ctx, "/material/number_fraction", number_fraction)
        if np.any(number_fraction < 0.0):
            ctx.add("TMAT_E009", "/material/number_fraction", "number_fraction must be >= 0")
        nsum = float(np.sum(number_fraction))
        if not np.isfinite(nsum) or abs(nsum - 1.0) > 1.0e-8:
            ctx.add(
                "TMAT_E010",
                "/material/number_fraction",
                f"sum should be 1 within 1e-8, got {nsum:.16e}",
            )
        if n_species is not None and number_fraction.shape != (n_species,):
            ctx.add(
                "TMAT_E006",
                "/material/number_fraction",
                f"shape mismatch with /material/Z: expected ({n_species},), got {number_fraction.shape}",
            )

    species_name = _read_string_vector_dataset(
        ctx, h5, "/material/species_name", required=False, strict_required=False
    )
    if species_name is not None and n_species is not None and len(species_name) != n_species:
        ctx.add(
            "TMAT_E006",
            "/material/species_name",
            f"expected length {n_species}, got {len(species_name)}",
        )

    return n_species


def _validate_eos(ctx: ValidationContext, h5: h5py.File) -> None:
    eos = _get_group(ctx, h5, "/eos", required=False)
    if eos is None:
        return

    axis_order = _read_attr_string(ctx, eos, "/eos", "axis_order", required=True)
    if axis_order is not None and axis_order != "D,T":
        ctx.add("TMAT_E006", "/eos/@axis_order", f"expected 'D,T', got '{axis_order}'")

    primary_axis = _read_attr_string(
        ctx, eos, "/eos", "primary_density_axis", required=True
    )
    if primary_axis is not None and primary_axis != "ni_cm3":
        ctx.add(
            "TMAT_E006",
            "/eos/@primary_density_axis",
            f"expected 'ni_cm3', got '{primary_axis}'",
        )

    _get_group(ctx, h5, "/eos/grid", required=True)
    _get_group(ctx, h5, "/eos/fields", required=True)

    rho = _read_float64_dataset(ctx, h5, "/eos/grid/ni_cm3", rank=1, required=True)
    temp = _read_float64_dataset(ctx, h5, "/eos/grid/temperature_eV", rank=1, required=True)
    if rho is not None:
        _check_positive_axis(ctx, "/eos/grid/ni_cm3", rho)
    if temp is not None:
        _check_positive_axis(ctx, "/eos/grid/temperature_eV", temp)

    n_d = int(rho.size) if rho is not None else None
    n_t = int(temp.size) if temp is not None else None
    shape = (n_d, n_t) if (n_d is not None and n_t is not None) else None

    zbar = _read_float64_dataset(
        ctx, h5, "/eos/fields/zbar", rank=2, shape=shape, required=True
    )
    if zbar is not None:
        _check_finite(ctx, "/eos/fields/zbar", zbar)
        if np.any(zbar < 0.0):
            ctx.add("TMAT_E009", "/eos/fields/zbar", "zbar must be >= 0")

    for path in (
        "/eos/fields/P_i",
        "/eos/fields/P_e",
        "/eos/fields/e_i",
        "/eos/fields/e_e",
    ):
        arr = _read_float64_dataset(ctx, h5, path, rank=2, shape=shape, required=True)
        if arr is not None:
            _check_finite(ctx, path, arr)

    for path in ("/eos/fields/cv_i", "/eos/fields/cv_e"):
        arr = _read_float64_dataset(
            ctx,
            h5,
            path,
            rank=2,
            shape=shape,
            required=False,
            strict_required=ctx.strict,
        )
        if arr is not None:
            _check_finite(ctx, path, arr)
            if np.any(arr <= 0.0):
                ctx.add("TMAT_E009", path, "heat capacity values must be > 0")


def _validate_ionization(ctx: ValidationContext, h5: h5py.File) -> None:
    ionization = _get_group(ctx, h5, "/ionization", required=False)
    if ionization is None:
        return

    axis_order = _read_attr_string(
        ctx, ionization, "/ionization", "axis_order", required=True
    )
    if axis_order is not None and axis_order != "S,D,T":
        ctx.add(
            "TMAT_E006",
            "/ionization/@axis_order",
            f"expected 'S,D,T', got '{axis_order}'",
        )

    primary_axis = _read_attr_string(
        ctx, ionization, "/ionization", "primary_density_axis", required=True
    )
    if primary_axis is not None and primary_axis != "ni_cm3":
        ctx.add(
            "TMAT_E006",
            "/ionization/@primary_density_axis",
            f"expected 'ni_cm3', got '{primary_axis}'",
        )

    _get_group(ctx, h5, "/ionization/grid", required=True)
    _get_group(ctx, h5, "/ionization/fields", required=True)

    rho = _read_float64_dataset(
        ctx, h5, "/ionization/grid/ni_cm3", rank=1, required=True
    )
    temp = _read_float64_dataset(
        ctx, h5, "/ionization/grid/temperature_eV", rank=1, required=True
    )
    if rho is not None:
        _check_positive_axis(ctx, "/ionization/grid/ni_cm3", rho)
    if temp is not None:
        _check_positive_axis(ctx, "/ionization/grid/temperature_eV", temp)

    eos = h5.get("/eos")
    if isinstance(eos, h5py.Group):
        for ion_path, eos_path, ion_values in (
            ("/ionization/grid/ni_cm3", "/eos/grid/ni_cm3", rho),
            (
                "/ionization/grid/temperature_eV",
                "/eos/grid/temperature_eV",
                temp,
            ),
        ):
            eos_ds = h5.get(eos_path)
            if ion_values is None or not isinstance(eos_ds, h5py.Dataset):
                continue
            try:
                eos_values = np.asarray(eos_ds[...], dtype=np.float64)
            except Exception:  # pragma: no cover - validated by _validate_eos
                continue
            if ion_values.shape != eos_values.shape:
                ctx.add(
                    "TMAT_E006",
                    ion_path,
                    f"shape must match {eos_path}: expected {eos_values.shape}, "
                    f"got {ion_values.shape}",
                )
            elif not np.allclose(
                ion_values, eos_values, rtol=1.0e-12, atol=0.0
            ):
                ctx.add(
                    "TMAT_E010",
                    ion_path,
                    f"values must match {eos_path} within relative tolerance 1e-12",
                )

    stage_element = _read_int32_dataset(
        ctx, h5, "/ionization/stage_element", rank=1, required=True
    )
    stage_charge = _read_int32_dataset(
        ctx, h5, "/ionization/stage_charge", rank=1, required=True
    )
    if (
        stage_element is not None
        and stage_charge is not None
        and stage_element.shape != stage_charge.shape
    ):
        ctx.add(
            "TMAT_E006",
            "/ionization/stage_charge",
            "length must match /ionization/stage_element",
        )

    n_s = int(stage_element.size) if stage_element is not None else None
    n_d = int(rho.size) if rho is not None else None
    n_t = int(temp.size) if temp is not None else None
    shape = (n_s, n_d, n_t) if None not in (n_s, n_d, n_t) else None
    fractions = _read_float64_dataset(
        ctx,
        h5,
        "/ionization/fields/fractions",
        rank=3,
        shape=shape,
        required=True,
    )
    if fractions is not None:
        _check_finite(ctx, "/ionization/fields/fractions", fractions)
        if np.any((fractions < -1.0e-12) | (fractions > 1.2)):
            ctx.add(
                "TMAT_E009",
                "/ionization/fields/fractions",
                "values must be within [-1e-12, 1.2]",
            )

    if (
        fractions is not None
        and fractions.ndim == 3
        and stage_element is not None
        and stage_element.ndim == 1
        and fractions.shape[0] == stage_element.size
    ):
        for element in np.unique(stage_element):
            point_sums = np.sum(fractions[stage_element == element, :, :], axis=0)
            bad_count = int(
                np.count_nonzero((point_sums < 0.8) | (point_sums > 1.2))
            )
            if bad_count > 0:
                ctx.add(
                    "TMAT_E010",
                    "/ionization/fields/fractions",
                    f"stage sums for element {int(element)} must be within "
                    f"[0.8, 1.2] at every grid point; {bad_count} points violate",
                )


def _validate_opacity(ctx: ValidationContext, h5: h5py.File) -> None:
    opacity = _get_group(ctx, h5, "/opacity", required=False)
    if opacity is None:
        return

    axis_order = _read_attr_string(ctx, opacity, "/opacity", "axis_order", required=True)
    if axis_order is not None and axis_order != "G,D,T":
        ctx.add("TMAT_E006", "/opacity/@axis_order", f"expected 'G,D,T', got '{axis_order}'")

    primary_axis = _read_attr_string(
        ctx, opacity, "/opacity", "primary_density_axis", required=True
    )
    if primary_axis is not None and primary_axis != "ni_cm3":
        ctx.add(
            "TMAT_E006",
            "/opacity/@primary_density_axis",
            f"expected 'ni_cm3', got '{primary_axis}'",
        )

    is_lte = _read_attr_int32_scalar(
        ctx, opacity, "/opacity", "is_lte", required=True, strict_required=ctx.strict
    )
    if is_lte is not None and is_lte not in (0, 1):
        ctx.add("TMAT_E009", "/opacity/@is_lte", f"expected 0 or 1, got {is_lte}")

    _get_group(ctx, h5, "/opacity/grid", required=True)
    _get_group(ctx, h5, "/opacity/fields", required=True)

    rho = _read_float64_dataset(ctx, h5, "/opacity/grid/ni_cm3", rank=1, required=True)
    temp = _read_float64_dataset(
        ctx, h5, "/opacity/grid/temperature_eV", rank=1, required=True
    )
    bounds = _read_float64_dataset(
        ctx, h5, "/opacity/grid/group_bounds_eV", rank=1, required=True
    )

    if rho is not None:
        _check_positive_axis(ctx, "/opacity/grid/ni_cm3", rho)
    if temp is not None:
        _check_positive_axis(ctx, "/opacity/grid/temperature_eV", temp)
    if bounds is not None:
        _check_group_bounds_axis(ctx, "/opacity/grid/group_bounds_eV", bounds)

    n_d = int(rho.size) if rho is not None else None
    n_t = int(temp.size) if temp is not None else None
    n_g = int(bounds.size - 1) if bounds is not None and bounds.size >= 1 else None
    if n_g is not None and n_g < 1:
        ctx.add("TMAT_E006", "/opacity/grid/group_bounds_eV", "requires at least 2 bounds")

    shape = (n_g, n_d, n_t) if None not in (n_g, n_d, n_t) else None
    for path in (
        "/opacity/fields/kappa_R",
        "/opacity/fields/kappa_PA",
        "/opacity/fields/kappa_PE",
    ):
        arr = _read_float64_dataset(ctx, h5, path, rank=3, shape=shape, required=True)
        if arr is not None:
            _check_finite(ctx, path, arr)
            if np.any(arr < 0.0):
                ctx.add("TMAT_E009", path, "opacity values must be >= 0")


def _validate_provenance(ctx: ValidationContext, h5: h5py.File) -> None:
    provenance = _get_group(
        ctx, h5, "/provenance", required=False, strict_required=ctx.strict
    )
    if provenance is None:
        return

    _read_string_scalar_dataset(
        ctx,
        h5,
        "/provenance/source_format",
        required=False,
        strict_required=ctx.strict,
    )
    source_files = _read_string_vector_dataset(
        ctx,
        h5,
        "/provenance/source_files",
        required=False,
        strict_required=ctx.strict,
    )
    if source_files is not None and len(source_files) < 1:
        ctx.add("TMAT_E006", "/provenance/source_files", "must have at least one entry")

    source_sha = _read_string_vector_dataset(
        ctx, h5, "/provenance/source_sha256", required=False, strict_required=False
    )
    if source_sha is not None and source_files is not None and len(source_sha) != len(source_files):
        ctx.add(
            "TMAT_E006",
            "/provenance/source_sha256",
            "length must match /provenance/source_files",
        )

    _read_string_scalar_dataset(
        ctx,
        h5,
        "/provenance/generator_name",
        required=False,
        strict_required=ctx.strict,
    )
    _read_string_scalar_dataset(
        ctx,
        h5,
        "/provenance/generator_version",
        required=False,
        strict_required=ctx.strict,
    )
    _read_string_scalar_dataset(
        ctx, h5, "/provenance/generator_git_commit", required=False, strict_required=False
    )
    _read_string_scalar_dataset(
        ctx, h5, "/provenance/command_line", required=False, strict_required=False
    )
    _read_string_scalar_dataset(ctx, h5, "/provenance/notes", required=False, strict_required=False)


def _validate_extensions(
    ctx: ValidationContext, h5: h5py.File, required_features: list[str]
) -> None:
    if not required_features:
        return

    for token in required_features:
        match = FEATURE_RE.match(token)
        if match is None:
            continue
        name = match.group(1)
        major = int(match.group(2))
        feature_path = f"/extensions/{name}/v{major}"
        obj = h5.get(feature_path)
        if obj is None:
            ctx.add("TMAT_E001", feature_path, f"required feature path for '{token}' is missing")
            continue
        if not isinstance(obj, h5py.Group):
            ctx.add("TMAT_E005", feature_path, "required feature path must be a group")


def _validate_dataset_filters(ctx: ValidationContext, h5: h5py.File) -> None:
    def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        path = f"/{name}" if name else "/"
        if obj.compression not in ALLOWED_COMPRESSION:
            ctx.add(
                "TMAT_E012",
                path,
                f"unsupported compression '{obj.compression}' (supported: none, gzip)",
            )
        plist = obj.id.get_create_plist()
        for i in range(plist.get_nfilters()):
            info = plist.get_filter(i)
            filter_id = int(info[0])
            if filter_id not in ALLOWED_FILTER_IDS:
                ctx.add(
                    "TMAT_E012",
                    path,
                    f"unsupported HDF5 filter id {filter_id}",
                )

    h5.visititems(visitor)


def validate(filepath: str, strict: bool = False) -> list[str]:
    """Validate a TMAT-H5 file.

    Parameters
    ----------
    filepath:
        Path to a `.tmat.h5` file.
    strict:
        If True, SHOULD items are also required where applicable.

    Returns
    -------
    list[str]
        List of validation violations. Empty list means valid.
    """

    ctx = ValidationContext(strict=bool(strict))
    try:
        _ensure_h5py()
        with h5py.File(filepath, "r") as h5:
            _validate_dataset_filters(ctx, h5)
            required_features = _validate_root(ctx, h5)
            _validate_material(ctx, h5)

            eos_obj = h5.get("/eos")
            opacity_obj = h5.get("/opacity")
            has_eos = isinstance(eos_obj, h5py.Group)
            has_opacity = isinstance(opacity_obj, h5py.Group)
            if not has_eos and not has_opacity:
                ctx.add("TMAT_E013", "/", "at least one of /eos or /opacity must exist")

            if "/eos" in h5:
                _validate_eos(ctx, h5)
            if "/ionization" in h5:
                _validate_ionization(ctx, h5)
            if "/opacity" in h5:
                _validate_opacity(ctx, h5)

            _validate_provenance(ctx, h5)
            _validate_extensions(ctx, h5, required_features)
    except OSError as exc:
        return [f"TMAT_E001 /: failed to open HDF5 file: {exc}"]
    except RuntimeError as exc:
        return [f"TMAT_E001 /: {exc}"]

    return ctx.violations


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate TENRYU TMAT-H5 files against spec v1.0."
    )
    parser.add_argument("filepath", help="Path to .tmat.h5 file")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict mode (SHOULD fields are treated as required).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    violations = validate(args.filepath, strict=args.strict)
    if violations:
        for violation in violations:
            print(violation, file=sys.stderr)
        return 1
    print(f"VALID: {args.filepath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
