#!/usr/bin/env python3
"""TENRYU reproducibility comparison and metadata utility."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import h5py


ARCH_ORDER = ("sm_80", "sm_89", "sm_90")
DEFAULT_N_OPS = 1_000_000
DEFAULT_TOLERANCE_C = 16.0
DOUBLE_EPSILON = sys.float_info.epsilon
NUMERIC_KINDS = frozenset("biuf")
REDUCTION_MODES = ("atomicAdd", "Kahan", "pairwise")


def normalize_sm_arch(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    if text.startswith("sm_"):
        return text
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return f"sm_{digits}"
    return text


def bool_from_text(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value!r}")


def tolerance_for(n_ops: int, tolerance_c: float = DEFAULT_TOLERANCE_C) -> float:
    return float(tolerance_c) * DOUBLE_EPSILON * max(0, int(n_ops))


def compare_files_bytewise(path_a: Path, path_b: Path, chunk_size: int = 1024 * 1024) -> dict[str, object]:
    size_a = path_a.stat().st_size
    size_b = path_b.stat().st_size
    first_diff: int | None = None
    diff_bytes = 0
    offset = 0

    with path_a.open("rb") as stream_a, path_b.open("rb") as stream_b:
        while True:
            chunk_a = stream_a.read(chunk_size)
            chunk_b = stream_b.read(chunk_size)
            if not chunk_a and not chunk_b:
                break

            common = min(len(chunk_a), len(chunk_b))
            if chunk_a[:common] != chunk_b[:common]:
                for idx, (byte_a, byte_b) in enumerate(zip(chunk_a, chunk_b)):
                    if byte_a != byte_b:
                        if first_diff is None:
                            first_diff = offset + idx
                        diff_bytes += 1
            if len(chunk_a) != len(chunk_b):
                if first_diff is None:
                    first_diff = offset + common
                diff_bytes += abs(len(chunk_a) - len(chunk_b))

            offset += max(len(chunk_a), len(chunk_b))

    status = "PASS" if first_diff is None and size_a == size_b else "FAIL"
    return {
        "mode": "same_arch_bitwise",
        "status": status,
        "path_a": str(path_a),
        "path_b": str(path_b),
        "size_a": size_a,
        "size_b": size_b,
        "first_diff_offset": first_diff,
        "diff_size_bytes": diff_bytes,
    }


def load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(value: str) -> dict[str, Any]:
    if value.lstrip().startswith("{"):
        return json.loads(value)
    candidate = Path(value)
    try:
        if candidate.exists():
            return load_json_file(candidate)
    except OSError:
        pass
    return json.loads(value)


def path_from_manifest_entry(entry: object) -> Path | None:
    if isinstance(entry, str):
        return Path(entry)
    if isinstance(entry, dict):
        for key in ("path", "hdf5", "hdf5_path", "output"):
            raw = entry.get(key)
            if raw:
                return Path(str(raw))
    return None


def ordered_arches(entries: dict[str, Path]) -> list[str]:
    order = {arch: idx for idx, arch in enumerate(ARCH_ORDER)}
    return sorted(entries, key=lambda arch: (order.get(arch, len(order)), arch))


def run_info_candidates(path: Path) -> Iterable[Path]:
    for directory in (path if path.is_dir() else path.parent, path.parent.parent):
        yield directory / "run_info.json"


def metadata_candidates(path: Path) -> Iterable[Path]:
    for directory in (path if path.is_dir() else path.parent, path.parent.parent):
        yield directory / "cross_arch_metadata.json"


def read_n_ops(path: Path | None, explicit_n_ops: int | None = None) -> int:
    if explicit_n_ops is not None:
        return int(explicit_n_ops)
    if path is None:
        return DEFAULT_N_OPS
    for candidate in run_info_candidates(path):
        if not candidate.exists():
            continue
        try:
            payload = load_json_file(candidate)
        except (OSError, json.JSONDecodeError):
            continue
        for key in ("n_ops", "N_op", "operation_count", "op_count"):
            if key in payload:
                return int(payload[key])
        counts = payload.get("operation_counts")
        if isinstance(counts, dict) and "total" in counts:
            return int(counts["total"])
    return DEFAULT_N_OPS


def detect_sm_arch() -> str:
    for name in ("TENRYU_SM_ARCH", "CUDA_SM_ARCH", "SM_ARCH"):
        if os.environ.get(name):
            return normalize_sm_arch(os.environ[name])

    cuda_arches = os.environ.get("CMAKE_CUDA_ARCHITECTURES")
    if cuda_arches:
        pieces = [piece for piece in cuda_arches.replace(",", ";").split(";") if piece]
        if len(pieces) == 1:
            return normalize_sm_arch(pieces[0])

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"

    first = result.stdout.splitlines()[0].strip() if result.stdout.splitlines() else ""
    return normalize_sm_arch(first.replace(".", ""))


def build_metadata(
    run_dir: Path,
    sm_arch: str | None,
    fmad_mode: bool,
    reduction_mode: str,
    n_ops: int,
    extra: list[str],
) -> dict[str, object]:
    if reduction_mode not in REDUCTION_MODES:
        raise ValueError(f"invalid reduction_mode {reduction_mode!r}; expected one of {', '.join(REDUCTION_MODES)}")
    payload: dict[str, object] = {
        "sm_arch": normalize_sm_arch(sm_arch) if sm_arch else detect_sm_arch(),
        "fmad_mode": bool(fmad_mode),
        "reduction_mode": reduction_mode,
        "n_ops": int(n_ops),
        "run_dir": str(run_dir),
        "generated_by": "tools/validation/reproducibility_check.py",
    }
    for item in extra:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"metadata entries must be key=value, got {item!r}")
        payload[key] = value
    return payload


def load_or_default_metadata(arch: str, path: Path, n_ops: int) -> dict[str, object]:
    for candidate in metadata_candidates(path):
        if candidate.exists():
            try:
                payload = load_json_file(candidate)
            except (OSError, json.JSONDecodeError):
                continue
            payload.setdefault("sm_arch", arch)
            payload.setdefault("fmad_mode", True)
            payload.setdefault("reduction_mode", "atomicAdd")
            payload.setdefault("n_ops", n_ops)
            return payload
    return {
        "sm_arch": arch,
        "fmad_mode": True,
        "reduction_mode": "atomicAdd",
        "n_ops": int(n_ops),
        "source": "defaulted_missing_cross_arch_metadata_json",
    }


def numeric_dataset_paths(handle: h5py.File) -> dict[str, tuple[tuple[int, ...], str]]:
    out: dict[str, tuple[tuple[int, ...], str]] = {}

    def visit(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset) and obj.dtype.kind in NUMERIC_KINDS:
            out["/" + name] = (tuple(int(dim) for dim in obj.shape), obj.dtype.str)

    handle.visititems(visit)
    return out


def iter_slices(shape: tuple[int, ...], target_elements: int = 65536) -> Iterable[object]:
    if not shape:
        yield ()
        return
    row_elements = 1
    for dim in shape[1:]:
        row_elements *= max(1, int(dim))
    rows_per_chunk = max(1, target_elements // row_elements)
    for start in range(0, shape[0], rows_per_chunk):
        stop = min(shape[0], start + rows_per_chunk)
        yield (slice(start, stop),) + tuple(slice(None) for _ in shape[1:])


def flat_values(value: object) -> Iterable[object]:
    flat = getattr(value, "flat", None)
    if flat is not None:
        return flat
    return (value,)


def relative_error(reference: object, candidate: object) -> float:
    ref = float(reference)
    other = float(candidate)
    if math.isnan(ref) or math.isnan(other):
        return 0.0 if math.isnan(ref) and math.isnan(other) else math.inf
    if math.isinf(ref) or math.isinf(other):
        return 0.0 if ref == other else math.inf
    diff = abs(other - ref)
    if diff == 0.0:
        return 0.0
    return diff / max(abs(ref), 1.0e-300)


def dataset_max_relative_error(ref_dataset: h5py.Dataset, other_dataset: h5py.Dataset) -> float:
    worst = 0.0
    for selection in iter_slices(tuple(int(dim) for dim in ref_dataset.shape)):
        ref_values = flat_values(ref_dataset[selection])
        other_values = flat_values(other_dataset[selection])
        for ref_value, other_value in zip(ref_values, other_values):
            worst = max(worst, relative_error(ref_value, other_value))
    return worst


def compare_arch_outputs(
    entries: dict[str, Path],
    reference_arch: str | None,
    tolerance: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    arches = ordered_arches(entries)
    ref_arch = reference_arch if reference_arch in entries else ("sm_80" if "sm_80" in entries else arches[0])
    field_results: list[dict[str, object]] = []
    structural_failures: list[dict[str, object]] = []

    with h5py.File(entries[ref_arch], "r") as ref_handle:
        ref_fields = numeric_dataset_paths(ref_handle)
        for arch in arches:
            if arch == ref_arch:
                continue
            with h5py.File(entries[arch], "r") as other_handle:
                other_fields = numeric_dataset_paths(other_handle)
                for field, (shape, dtype) in ref_fields.items():
                    if field not in other_fields:
                        structural_failures.append({"arch": arch, "field": field, "reason": "missing_dataset"})
                        continue
                    other_shape, other_dtype = other_fields[field]
                    if other_shape != shape:
                        structural_failures.append(
                            {
                                "arch": arch,
                                "field": field,
                                "reason": "shape_mismatch",
                                "reference_shape": shape,
                                "candidate_shape": other_shape,
                            }
                        )
                        continue
                    if other_dtype != dtype:
                        structural_failures.append(
                            {
                                "arch": arch,
                                "field": field,
                                "reason": "dtype_mismatch",
                                "reference_dtype": dtype,
                                "candidate_dtype": other_dtype,
                            }
                        )
                        continue
                    max_error = dataset_max_relative_error(ref_handle[field], other_handle[field])
                    field_results.append(
                        {
                            "arch": arch,
                            "field": field,
                            "max_relative_error": max_error,
                            "tolerance": tolerance,
                            "status": "PASS" if max_error <= tolerance else "FAIL",
                        }
                    )
                for field in sorted(set(other_fields) - set(ref_fields)):
                    structural_failures.append({"arch": arch, "field": field, "reason": "extra_dataset"})

    return field_results, structural_failures


def run_same_arch_bitwise(args: argparse.Namespace) -> int:
    result = compare_files_bytewise(Path(args.path_a), Path(args.path_b), args.chunk_size)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


def run_log_metadata(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    n_ops = read_n_ops(run_dir, args.n_ops)
    try:
        metadata = build_metadata(
            run_dir=run_dir,
            sm_arch=args.sm_arch,
            fmad_mode=args.fmad_mode,
            reduction_mode=args.reduction_mode,
            n_ops=n_ops,
            extra=args.metadata,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    out_path = run_dir / "cross_arch_metadata.json"
    out_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "path": str(out_path), "cross_arch_metadata": metadata}, indent=2, sort_keys=True))
    return 0


def run_cross_arch_tolerance(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    entries: dict[str, Path] = {}
    missing_arches: list[str] = []
    for arch in ARCH_ORDER:
        path = path_from_manifest_entry(manifest.get(arch))
        if path is not None and path.exists():
            entries[arch] = path
        else:
            missing_arches.append(arch)

    for arch, entry in manifest.items():
        if arch in ARCH_ORDER:
            continue
        path = path_from_manifest_entry(entry)
        if path is not None and path.exists():
            entries[arch] = path

    arches = ordered_arches(entries)
    n_ops = read_n_ops(entries[arches[0]] if arches else None, args.n_ops)
    metadata_by_arch = {arch: load_or_default_metadata(arch, path, n_ops) for arch, path in entries.items()}

    if len(entries) < 2:
        arch = arches[0] if arches else "none"
        metadata: object = metadata_by_arch.get(arch, {})
        result = {
            "mode": "cross_arch_tolerance",
            "status": "deferred_to_multi_arch_ci",
            "reason": f"single-arch {arch} detected" if len(entries) == 1 else "no architecture outputs detected",
            "available_arches": arches,
            "missing_arches": missing_arches,
            "cross_arch_metadata": metadata,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    tolerance = tolerance_for(n_ops, args.tolerance_c)
    try:
        field_results, structural_failures = compare_arch_outputs(entries, args.reference_arch, tolerance)
    except OSError as exc:
        result = {
            "mode": "cross_arch_tolerance",
            "status": "FAIL",
            "reason": f"HDF5 read failure: {exc}",
            "available_arches": arches,
            "cross_arch_metadata": metadata_by_arch,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1

    worst_case = max(field_results, key=lambda item: float(item["max_relative_error"]), default=None)
    has_error_failure = any(item["status"] == "FAIL" for item in field_results)
    status = "FAIL" if structural_failures or has_error_failure or not field_results else "PASS"
    result = {
        "mode": "cross_arch_tolerance",
        "status": status,
        "reference_arch": args.reference_arch if args.reference_arch in entries else ("sm_80" if "sm_80" in entries else arches[0]),
        "available_arches": arches,
        "missing_arches": missing_arches,
        "epsilon": DOUBLE_EPSILON,
        "n_ops": n_ops,
        "tolerance_c": args.tolerance_c,
        "tolerance": tolerance,
        "field_count": len(field_results),
        "worst_case": worst_case,
        "structural_failures": structural_failures,
        "cross_arch_metadata": metadata_by_arch,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    same = subparsers.add_parser("same_arch_bitwise", help="byte-compare two HDF5 files")
    same.add_argument("path_a")
    same.add_argument("path_b")
    same.add_argument("--chunk-size", type=int, default=1024 * 1024)
    same.set_defaults(func=run_same_arch_bitwise)

    cross = subparsers.add_parser("cross_arch_tolerance", help="compare a per-architecture HDF5 manifest")
    cross.add_argument("manifest", help="JSON file path or inline JSON object")
    cross.add_argument("--reference-arch")
    cross.add_argument("--n-ops", type=int)
    cross.add_argument("--tolerance-c", type=float, default=DEFAULT_TOLERANCE_C)
    cross.set_defaults(func=run_cross_arch_tolerance)

    metadata = subparsers.add_parser("log_metadata", help="write cross_arch_metadata.json")
    metadata.add_argument("run_dir")
    metadata.add_argument("--sm-arch")
    metadata.add_argument("--fmad-mode", type=bool_from_text, default=True)
    metadata.add_argument(
        "--reduction-mode",
        choices=REDUCTION_MODES,
        default=os.environ.get("TENRYU_REDUCTION_MODE", "atomicAdd"),
    )
    metadata.add_argument("--n-ops", type=int)
    metadata.add_argument("--metadata", action="append", default=[])
    metadata.set_defaults(func=run_log_metadata)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
