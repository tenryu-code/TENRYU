"""HDF5 readers for TENRYU 1D output directories."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py
except ModuleNotFoundError:
    h5py = None


SNAPSHOT_GLOB = "*_[0-9][0-9][0-9][0-9].h5"
SNAPSHOT_INDEX_RE = re.compile(r"_(\d{4})\.h5$")
GEOMETRIES = {"planar", "cylindrical", "spherical"}


class SnapshotFormatError(RuntimeError):
    """Raised when a TENRYU HDF5 file does not match the expected schema."""


def _ensure_h5py() -> None:
    if h5py is None:
        raise RuntimeError("h5py is required. Install dependency: pip install h5py")


def _snapshot_index(path: Path) -> int:
    match = SNAPSHOT_INDEX_RE.search(path.name)
    if match is None:
        raise SnapshotFormatError(f"snapshot file does not end with _NNNN.h5: {path}")
    return int(match.group(1))


def _as_scalar(value: Any, name: str) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    raise SnapshotFormatError(f"{name} must be scalar, got shape {arr.shape}")


def _attr_scalar(h5: Any, name: str) -> Any:
    if name not in h5.attrs:
        raise SnapshotFormatError(f"missing required root attribute '{name}'")
    return _as_scalar(h5.attrs[name], f"root attribute '{name}'")


def _require_dataset(h5: Any, name: str) -> Any:
    obj = h5.get(name)
    if obj is None:
        raise SnapshotFormatError(f"missing required dataset '{name}'")
    if not isinstance(obj, h5py.Dataset):
        raise SnapshotFormatError(f"expected dataset at '{name}'")
    return obj


def _optional_dataset(h5: Any, name: str) -> np.ndarray | None:
    obj = h5.get(name)
    if obj is None:
        return None
    if not isinstance(obj, h5py.Dataset):
        raise SnapshotFormatError(f"expected dataset at '{name}'")
    return np.asarray(obj[...])


def _decode_scalar_text(value: Any) -> str:
    item = _as_scalar(value, "scalar text")
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _read_geometry(h5: Any) -> tuple[str, str]:
    obj = h5.get("metadata/frozen_config")
    if obj is None:
        return "spherical", "default"
    if not isinstance(obj, h5py.Dataset):
        raise SnapshotFormatError("expected dataset at 'metadata/frozen_config'")

    text = _decode_scalar_text(obj[()])
    config = json.loads(text)
    geometry = config.get("mesh", {}).get("geometry_1d")
    if geometry is None:
        return "spherical", "default"
    if geometry not in GEOMETRIES:
        raise SnapshotFormatError(f"unsupported mesh.geometry_1d: {geometry}")
    return geometry, "frozen_config"


@dataclass
class RunDir:
    """Discovered TENRYU output directory."""

    path: Path
    case: str
    results_dir: Path
    snapshot_paths: list[Path]
    history_path: Path | None
    run_info: dict[str, Any] | None
    _time_cache: list[tuple[Path, float]] | None = dataclass_field(
        default=None, init=False, repr=False
    )

    @staticmethod
    def find(path: str | Path) -> "RunDir":
        candidate = Path(path)
        if not candidate.exists():
            raise FileNotFoundError(f"path not found: {candidate}")
        if not candidate.is_dir():
            raise NotADirectoryError(f"path is not a directory: {candidate}")

        if (candidate / "results").is_dir():
            outdir = candidate
            results_dir = candidate / "results"
        else:
            results_dir = candidate
            outdir = candidate.parent if candidate.name == "results" else candidate

        snapshot_paths = _collect_snapshot_paths(results_dir)
        case = _infer_case(snapshot_paths[0])
        history_path = _find_history_path(results_dir, case)
        run_info = _read_run_info(outdir)
        return RunDir(
            path=outdir,
            case=case,
            results_dir=results_dir,
            snapshot_paths=snapshot_paths,
            history_path=history_path,
            run_info=run_info,
        )

    def snapshot_times(self) -> list[tuple[Path, float]]:
        if self._time_cache is None:
            _ensure_h5py()
            pairs: list[tuple[Path, float]] = []
            for path in self.snapshot_paths:
                with h5py.File(path, "r") as h5:
                    pairs.append((path, float(_attr_scalar(h5, "t"))))
            self._time_cache = pairs
        return self._time_cache


def _collect_snapshot_paths(results_dir: Path) -> list[Path]:
    if not results_dir.exists():
        raise FileNotFoundError(f"results directory not found: {results_dir}")
    if not results_dir.is_dir():
        raise NotADirectoryError(f"results path is not a directory: {results_dir}")

    paths: list[Path] = []
    for path in results_dir.glob(SNAPSHOT_GLOB):
        if not path.is_file():
            continue
        name = path.name
        if "_ckpt_" in name or "_history" in name:
            continue
        if SNAPSHOT_INDEX_RE.search(name) is None:
            continue
        paths.append(path)

    paths.sort(key=lambda p: (_snapshot_index(p), p.name))
    if not paths:
        raise FileNotFoundError(f"no snapshot files matched '{SNAPSHOT_GLOB}' in {results_dir}")
    return paths


def _infer_case(path: Path) -> str:
    match = SNAPSHOT_INDEX_RE.search(path.name)
    if match is None:
        return path.stem
    return path.name[: match.start()]


def _find_history_path(results_dir: Path, case: str) -> Path | None:
    preferred = results_dir / f"{case}_history.h5"
    if preferred.is_file():
        return preferred
    matches = sorted(p for p in results_dir.glob("*_history.h5") if p.is_file())
    return matches[0] if matches else None


def _read_run_info(outdir: Path) -> dict[str, Any] | None:
    path = outdir / "run_info.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Snapshot:
    """One TENRYU results snapshot."""

    path: Path
    t: float
    dt: float
    step: int
    n_cells: int
    n_groups: int
    n_materials: int
    x_r: np.ndarray
    v_r: np.ndarray
    material_id: np.ndarray
    hydro: dict[str, np.ndarray]
    rad_energy_density: np.ndarray | None
    laser_deposited_power: np.ndarray | None
    laser_absorption_fraction: float | None
    group_bounds_eV: np.ndarray | None
    geometry: str
    geometry_source: str
    _r_centers: np.ndarray | None = dataclass_field(default=None, init=False, repr=False)
    _dr: np.ndarray | None = dataclass_field(default=None, init=False, repr=False)

    @staticmethod
    def load(path: str | Path, fields: list[str] | None = None) -> "Snapshot":
        _ensure_h5py()
        snapshot_path = Path(path)
        with h5py.File(snapshot_path, "r") as h5:
            t = float(_attr_scalar(h5, "t"))
            dt = float(_attr_scalar(h5, "dt"))
            step = int(_attr_scalar(h5, "step"))
            n_cells = int(_attr_scalar(h5, "n_cells"))
            n_groups = int(_attr_scalar(h5, "n_groups"))
            n_materials = int(_attr_scalar(h5, "n_materials"))

            x_r = np.asarray(_require_dataset(h5, "mesh/x_r")[...], dtype=np.float64)
            v_r = np.asarray(_require_dataset(h5, "mesh/v_r")[...], dtype=np.float64)
            material_id = np.asarray(
                _require_dataset(h5, "mesh/cell_material_id")[...], dtype=np.int64
            )
            hydro = _read_hydro_group(h5)
            rad_energy_density = _optional_dataset(h5, "radiation/energy_density")
            if rad_energy_density is not None:
                rad_energy_density = np.asarray(rad_energy_density, dtype=np.float64)
            laser_deposited_power = _optional_dataset(h5, "laser/deposited_power")
            if laser_deposited_power is not None:
                laser_deposited_power = np.asarray(laser_deposited_power, dtype=np.float64)
            laser_absorption_fraction = _read_optional_scalar(
                h5, "laser/absorption_fraction"
            )
            group_bounds_eV = _optional_dataset(h5, "metadata/group_bounds_eV")
            if group_bounds_eV is not None:
                group_bounds_eV = np.asarray(group_bounds_eV, dtype=np.float64)
            geometry, geometry_source = _read_geometry(h5)

        _validate_snapshot_arrays(snapshot_path, x_r, v_r, material_id, n_cells)
        return Snapshot(
            path=snapshot_path,
            t=t,
            dt=dt,
            step=step,
            n_cells=n_cells,
            n_groups=n_groups,
            n_materials=n_materials,
            x_r=x_r,
            v_r=v_r,
            material_id=material_id,
            hydro=hydro,
            rad_energy_density=rad_energy_density,
            laser_deposited_power=laser_deposited_power,
            laser_absorption_fraction=laser_absorption_fraction,
            group_bounds_eV=group_bounds_eV,
            geometry=geometry,
            geometry_source=geometry_source,
        )

    def field(self, name: str) -> np.ndarray | None:
        if name.startswith("hydro/"):
            name = name[len("hydro/") :]
        return self.hydro.get(name)

    @property
    def r_centers(self) -> np.ndarray:
        if self._r_centers is None:
            self._r_centers = 0.5 * (self.x_r[:-1] + self.x_r[1:])
        return self._r_centers

    @property
    def dr(self) -> np.ndarray:
        if self._dr is None:
            self._dr = np.diff(self.x_r)
        return self._dr


def _read_hydro_group(h5: Any) -> dict[str, np.ndarray]:
    obj = h5.get("hydro")
    if obj is None:
        return {}
    if not isinstance(obj, h5py.Group):
        raise SnapshotFormatError("expected group at 'hydro'")

    hydro: dict[str, np.ndarray] = {}

    def visitor(name: str, item: Any) -> None:
        if isinstance(item, h5py.Dataset):
            hydro[name] = np.asarray(item[...])

    obj.visititems(visitor)
    return hydro


def _read_optional_scalar(h5: Any, name: str) -> float | None:
    obj = h5.get(name)
    if obj is None:
        return None
    if not isinstance(obj, h5py.Dataset):
        raise SnapshotFormatError(f"expected dataset at '{name}'")
    return float(_as_scalar(obj[()], name))


def _validate_snapshot_arrays(
    path: Path,
    x_r: np.ndarray,
    v_r: np.ndarray,
    material_id: np.ndarray,
    n_cells: int,
) -> None:
    if x_r.ndim != 1:
        raise SnapshotFormatError(f"{path.name}: mesh/x_r must be rank-1")
    if v_r.ndim != 1:
        raise SnapshotFormatError(f"{path.name}: mesh/v_r must be rank-1")
    if material_id.ndim != 1:
        raise SnapshotFormatError(f"{path.name}: mesh/cell_material_id must be rank-1")
    if x_r.size != n_cells + 1:
        raise SnapshotFormatError(
            f"{path.name}: mesh/x_r has {x_r.size} nodes, expected {n_cells + 1}"
        )
    if v_r.size != x_r.size:
        raise SnapshotFormatError(
            f"{path.name}: mesh/v_r has {v_r.size} nodes, expected {x_r.size}"
        )
    if material_id.size != n_cells:
        raise SnapshotFormatError(
            f"{path.name}: mesh/cell_material_id has {material_id.size} cells, expected {n_cells}"
        )


@dataclass
class History:
    """Generic TENRYU history ledger reader."""

    path: Path
    t: np.ndarray
    cycle: np.ndarray
    dt: np.ndarray
    series: dict[str, np.ndarray]
    static: dict[str, np.ndarray]

    @staticmethod
    def load(path: str | Path) -> "History":
        _ensure_h5py()
        history_path = Path(path)
        with h5py.File(history_path, "r") as h5:
            t = np.asarray(_require_dataset(h5, "t")[...])
            cycle = np.asarray(_require_dataset(h5, "cycle")[...])
            dt = np.asarray(_require_dataset(h5, "dt")[...])
            n_records = len(t)
            series: dict[str, np.ndarray] = {}
            static: dict[str, np.ndarray] = {}

            def visitor(name: str, item: Any) -> None:
                if not isinstance(item, h5py.Dataset):
                    return
                if name in {"t", "cycle", "dt"}:
                    return
                arr = np.asarray(item[...])
                # String step-series like dt_winner exist and are not plottable numeric series.
                if np.issubdtype(arr.dtype, np.number) and arr.ndim >= 1 and arr.shape[0] == n_records:
                    series[name] = arr
                else:
                    static[name] = arr

            h5.visititems(visitor)

        return History(
            path=history_path,
            t=t,
            cycle=cycle,
            dt=dt,
            series=series,
            static=static,
        )

    def get(self, name: str) -> np.ndarray | None:
        return self.series.get(name)

    def group(self, prefix: str) -> dict[str, np.ndarray]:
        return {name: arr for name, arr in self.series.items() if name.startswith(prefix)}


def pick_snapshots(
    run: RunDir,
    at: float | str | None = None,
    index: int | None = None,
    last: bool = False,
    every: int | None = None,
) -> list[Path]:
    selectors = int(at is not None) + int(index is not None) + int(last) + int(every is not None)
    if selectors > 1:
        raise ValueError("choose only one of at, index, last, or every")

    if every is not None:
        if every <= 0:
            raise ValueError("every must be > 0")
        return run.snapshot_paths[::every]
    if at is not None:
        target = _parse_time_seconds(at)
        pairs = run.snapshot_times()
        if not pairs:
            raise FileNotFoundError(f"no snapshots in {run.results_dir}")
        path, _ = min(pairs, key=lambda item: abs(item[1] - target))
        return [path]
    if index is not None:
        for path in run.snapshot_paths:
            if _snapshot_index(path) == index:
                return [path]
        raise IndexError(f"snapshot index {index:04d} not found in {run.results_dir}")
    if last:
        return [run.snapshot_paths[-1]]
    return list(run.snapshot_paths)


def _parse_time_seconds(value: float | str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    match = re.match(
        r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(ns|us|ps|s)?$",
        text,
    )
    if match is None:
        raise ValueError(f"invalid time selector: {value}")
    number = float(match.group(1))
    suffix = match.group(2) or ""
    scale = {"": 1.0, "s": 1.0, "ns": 1.0e-9, "us": 1.0e-6, "ps": 1.0e-12}[suffix]
    return number * scale
