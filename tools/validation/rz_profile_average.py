"""2D RZ to 1D z-profile reduction by volume-weighted radial averaging.

For r-invariant z-slab radiative-shock verification, this module reduces
2D RZ cell-centered field arrays to 1D z-profiles by integrating along the
radial direction with cell-volume weighting.  It also computes the
radial-invariance diagnostic

    epsilon_r = max|q_ij - avg_r(q_j)| / max|avg_r(q_j)|

used by the I1-I7 production verification harnesses.

The validation directory is used as a flat Python path in TENRYU.  Import this
module with, for example:

    sys.path.insert(0, "tools/validation")
    from rz_profile_average import radial_average
"""

from __future__ import annotations

from typing import Any

import numpy as np


TINY = 1.0e-300
EPSILON_FLOOR = 0.0


def _as_float_array(value: Any) -> np.ndarray:
    try:
        value = value[()]
    except Exception:
        pass
    return np.asarray(value, dtype=float)


def _snapshot_value(snapshot: Any, *names: str) -> Any:
    for name in names:
        if isinstance(snapshot, dict) and name in snapshot:
            return snapshot[name]
        if "/" not in name and hasattr(snapshot, name):
            return getattr(snapshot, name)
        try:
            if name in snapshot:
                return snapshot[name]
        except Exception:
            pass
        attrs = getattr(snapshot, "attrs", None)
        if attrs is not None and name in attrs:
            return attrs[name]
    raise KeyError(f"snapshot does not expose any of: {', '.join(names)}")


def _snapshot_int(snapshot: Any, name: str) -> int | None:
    try:
        value = _snapshot_value(snapshot, name)
    except KeyError:
        return None
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return None
    return int(arr[0])


def _cell_shape_from_snapshot(snapshot: Any) -> tuple[int, int] | None:
    nr = _snapshot_int(snapshot, "nr")
    nz = _snapshot_int(snapshot, "nz")
    if nr is not None and nz is not None:
        return nr, nz
    try:
        r_nodes = _as_float_array(_snapshot_value(snapshot, "r_nodes", "mesh/x_r"))
        z_nodes = _as_float_array(_snapshot_value(snapshot, "z_nodes", "mesh/x_z"))
    except KeyError:
        return None
    if r_nodes.ndim == 2 and z_nodes.ndim == 2 and r_nodes.shape == z_nodes.shape:
        return r_nodes.shape[0] - 1, r_nodes.shape[1] - 1
    if r_nodes.ndim == 1 and z_nodes.ndim == 1:
        return r_nodes.size - 1, z_nodes.size - 1
    return None


def _reshape_cell_array(values: Any, expected_shape: tuple[int, int] | None, name: str) -> np.ndarray:
    arr = _as_float_array(values)
    if expected_shape is not None:
        if arr.shape == expected_shape:
            return arr
        if arr.size == expected_shape[0] * expected_shape[1]:
            return arr.reshape(expected_shape)
        raise ValueError(f"{name}: expected {expected_shape} cells, got shape {arr.shape}")
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[:, :, 0]
    raise ValueError(f"{name}: cannot infer 2D cell shape from {arr.shape}")


def _reshape_node_array(values: Any, nr: int | None, nz: int | None, name: str) -> np.ndarray:
    arr = _as_float_array(values)
    if arr.ndim == 1:
        return arr
    if nr is not None and nz is not None:
        expected = (nr + 1, nz + 1)
        if arr.shape == expected:
            return arr
        if arr.size == expected[0] * expected[1]:
            return arr.reshape(expected)
        raise ValueError(f"{name}: expected node shape {expected}, got {arr.shape}")
    if arr.ndim == 2:
        return arr
    raise ValueError(f"{name}: cannot infer 2D node shape from {arr.shape}")


def radial_average(field_2d: np.ndarray, vol_2d: np.ndarray, axis_r: int = 0) -> np.ndarray:
    """Return the volume-weighted radial average of a 2D RZ field.

    Parameters
    ----------
    field_2d : np.ndarray
        Cell-centered field with shape ``(n_r, n_z)`` when ``axis_r == 0`` or
        ``(n_z, n_r)`` when ``axis_r == 1``.
    vol_2d : np.ndarray
        Cell volumes in cm^3 with the same shape as ``field_2d``.
    axis_r : int
        Axis index corresponding to the radial dimension.  Must be 0 or 1.

    Returns
    -------
    np.ndarray
        The z-profile after volume-weighted radial averaging.
    """
    field = np.asarray(field_2d, dtype=float)
    vol = np.asarray(vol_2d, dtype=float)
    if field.ndim != 2 or vol.ndim != 2:
        raise ValueError("field_2d and vol_2d must both be 2D arrays")
    if field.shape != vol.shape:
        raise ValueError(f"field_2d shape {field.shape} does not match vol_2d shape {vol.shape}")
    if axis_r not in (0, 1):
        raise ValueError("axis_r must be 0 or 1")
    numer = np.sum(field * vol, axis=axis_r)
    denom = np.sum(vol, axis=axis_r) + TINY
    return numer / denom


def radial_invariance_metric(field_2d: np.ndarray, field_avg: np.ndarray, axis_r: int = 0) -> float:
    """Return the max-norm radial-invariance metric epsilon_r.

    Parameters
    ----------
    field_2d : np.ndarray
        Cell-centered field with one radial axis and one z axis.
    field_avg : np.ndarray
        Radial average returned by :func:`radial_average`.
    axis_r : int
        Axis index corresponding to the radial dimension.  Must be 0 or 1.

    Returns
    -------
    float
        ``max(abs(field_2d - field_avg)) / max(abs(field_avg))`` with a tiny
        denominator guard.
    """
    field = np.asarray(field_2d, dtype=float)
    avg = np.asarray(field_avg, dtype=float)
    if field.ndim != 2 or avg.ndim != 1:
        raise ValueError("field_2d must be 2D and field_avg must be 1D")
    if axis_r == 0:
        if field.shape[1] != avg.size:
            raise ValueError(f"field z-size {field.shape[1]} does not match average size {avg.size}")
        dev = field - avg[None, :]
    elif axis_r == 1:
        if field.shape[0] != avg.size:
            raise ValueError(f"field z-size {field.shape[0]} does not match average size {avg.size}")
        dev = field - avg[:, None]
    else:
        raise ValueError("axis_r must be 0 or 1")
    denom = float(np.max(np.abs(avg))) + TINY
    return float(max(EPSILON_FLOOR, np.max(np.abs(dev)) / denom))


def cell_volumes_from_hdf5(snapshot: Any) -> np.ndarray:
    """Extract or compute 2D RZ cell volumes from a TENRYU HDF5 snapshot.

    Parameters
    ----------
    snapshot : Any
        Snapshot object, dict, or open HDF5 handle exposing either ``vol`` /
        ``hydro/vol`` or mesh nodes ``r_nodes`` / ``z_nodes`` or
        ``mesh/x_r`` / ``mesh/x_z``.

    Returns
    -------
    np.ndarray
        Cell volumes in cm^3 with shape ``(n_r, n_z)``.
    """
    expected_shape = _cell_shape_from_snapshot(snapshot)
    try:
        return _reshape_cell_array(
            _snapshot_value(snapshot, "vol", "hydro/vol", "cell_volume", "cell_volumes"),
            expected_shape,
            "cell volume",
        )
    except KeyError:
        pass

    nr = expected_shape[0] if expected_shape is not None else None
    nz = expected_shape[1] if expected_shape is not None else None
    r_nodes = _reshape_node_array(_snapshot_value(snapshot, "r_nodes", "mesh/x_r"), nr, nz, "r nodes")
    z_nodes = _reshape_node_array(_snapshot_value(snapshot, "z_nodes", "mesh/x_z"), nr, nz, "z nodes")
    if r_nodes.ndim == 1 and z_nodes.ndim == 1:
        r_inner = r_nodes[:-1, None]
        r_outer = r_nodes[1:, None]
        dz = np.abs(np.diff(z_nodes))[None, :]
        return np.pi * np.abs(r_outer * r_outer - r_inner * r_inner) * dz
    if r_nodes.ndim != 2 or z_nodes.ndim != 2 or r_nodes.shape != z_nodes.shape:
        raise ValueError(f"r/z node shapes must match as 2D arrays, got {r_nodes.shape} and {z_nodes.shape}")
    r_inner = 0.5 * (r_nodes[:-1, :-1] + r_nodes[:-1, 1:])
    r_outer = 0.5 * (r_nodes[1:, :-1] + r_nodes[1:, 1:])
    z_lower = 0.5 * (z_nodes[:-1, :-1] + z_nodes[1:, :-1])
    z_upper = 0.5 * (z_nodes[:-1, 1:] + z_nodes[1:, 1:])
    return np.pi * np.abs(r_outer * r_outer - r_inner * r_inner) * np.abs(z_upper - z_lower)


def reduce_2d_rz_to_z_profile(field_2d: np.ndarray, snapshot: Any) -> dict[str, Any]:
    """Reduce a 2D RZ cell field to a z-profile and invariance diagnostic.

    Parameters
    ----------
    field_2d : np.ndarray
        Cell-centered field with shape ``(n_r, n_z)``.
    snapshot : Any
        Snapshot object, dict, or open HDF5 handle accepted by
        :func:`cell_volumes_from_hdf5`.

    Returns
    -------
    dict[str, Any]
        JSON-compatible metadata containing ``profile_z``, ``invariance``,
        ``n_r``, and ``n_z``.
    """
    field = np.asarray(field_2d, dtype=float)
    if field.ndim != 2:
        raise ValueError(f"field_2d must be 2D, got shape {field.shape}")
    vol = cell_volumes_from_hdf5(snapshot)
    if vol.shape != field.shape:
        raise ValueError(f"volume shape {vol.shape} does not match field shape {field.shape}")
    avg = radial_average(field, vol, axis_r=0)
    inv = radial_invariance_metric(field, avg, axis_r=0)
    return {"profile_z": avg, "invariance": inv, "n_r": int(field.shape[0]), "n_z": int(field.shape[1])}
