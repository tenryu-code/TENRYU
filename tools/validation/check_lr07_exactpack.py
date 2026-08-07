#!/usr/bin/env python3
"""Cross-check the in-house LR07-EDA reference against ExactPack when available."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


TINY = 1.0e-300
IDENTITY_MEDIAN_GATE = 1.0e-3
IDENTITY_P95_GATE = 1.0e-2


try:
    import exactpack.solvers.radshocks as radshocks

    EXACTPACK_AVAILABLE = True
except ImportError:
    radshocks = None
    EXACTPACK_AVAILABLE = False


def safe_float_token(value: float) -> str:
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def default_reference_path(mach: float) -> Path:
    repo = Path(__file__).resolve().parents[2]
    return repo / "tests" / "verification" / "data" / "lr07_eda_reference" / (
        f"lr07_eda_M{safe_float_token(mach)}.json"
    )


def read_reference(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def schema_version(ref: dict[str, Any]) -> Any:
    return ref.get("schema_version")


def is_schema_v2(ref: dict[str, Any]) -> bool:
    version = schema_version(ref)
    return version == 2 or str(version).strip() == "2" or str(version).strip().endswith(".v2")


def table_field(table: dict[str, Any], *names: str) -> np.ndarray:
    for name in names:
        if name in table:
            return np.asarray(table[name], dtype=float)
    raise KeyError(f"missing any reference table field from {names}")


def branch_labels(table: dict[str, Any], x: np.ndarray) -> np.ndarray:
    if "branch" in table:
        return np.asarray(table["branch"], dtype=str)
    return np.where(x <= 0.0, "upstream", "downstream")


def branch_identity_rel(
    x: np.ndarray,
    rho: np.ndarray,
    T: np.ndarray,
    F_ref: np.ndarray,
    branches: np.ndarray,
    branch: str,
    a_eV: float,
    c_light: float,
    kappa_R: float,
    *,
    branch_oriented_flux: bool,
) -> np.ndarray:
    mask = branches == branch
    if np.count_nonzero(mask) < 3:
        return np.asarray([math.inf], dtype=float)
    x_b = x[mask]
    rho_b = rho[mask]
    T_b = T[mask]
    F_b = F_ref[mask]
    order = np.argsort(x_b)
    x_b = x_b[order]
    rho_b = rho_b[order]
    T_b = T_b[order]
    F_b = F_b[order]
    E = a_eV * T_b**4
    D = c_light / (3.0 * kappa_R * np.maximum(rho_b, TINY))
    F_from_grad = -D * np.gradient(E, x_b)
    if branch_oriented_flux and branch == "downstream":
        F_from_grad = -F_from_grad
    rel = (F_b - F_from_grad) / np.maximum(np.abs(F_b) + np.abs(F_from_grad), TINY)
    interior = np.abs(rel[10:-10]) if rel.size > 20 else np.abs(rel)
    return interior


def summarize_identity_rels(rels: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for branch, values in rels.items():
        out[branch] = (float(np.median(values)), float(np.percentile(values, 95.0)))
    total_values = np.concatenate([rels["upstream"], rels["downstream"]])
    out["total"] = (float(np.median(total_values)), float(np.percentile(total_values, 95.0)))
    return out


def diffusion_identity_check(
    ref: dict[str, Any],
    *,
    flux_field: str,
    branch_oriented_flux: bool,
) -> dict[str, tuple[float, float]]:
    table = ref["table"]
    params = ref["parameters"]
    x = table_field(table, "x_cm")
    rho = table_field(table, "rho_g_per_cc")
    T = table_field(table, "T_eV")
    F_ref = table_field(table, flux_field, "F_rad_erg_cm2_s")
    branches = branch_labels(table, x)
    a_eV = float(params["a_eV_erg_cm3_eV4"])
    c_light = float(params["c_cm_per_s"])
    kappa_R = float(params["kappa_R_cm2_per_g"])

    rels = {
        branch: branch_identity_rel(
            x,
            rho,
            T,
            F_ref,
            branches,
            branch,
            a_eV,
            c_light,
            kappa_R,
            branch_oriented_flux=branch_oriented_flux,
        )
        for branch in ("upstream", "downstream")
    }
    return summarize_identity_rels(rels)


def in_house_diffusion_identity_check(ref: dict[str, Any]) -> dict[str, tuple[float, float]]:
    return diffusion_identity_check(
        ref,
        flux_field="F_rad_erg_per_cm2_s",
        branch_oriented_flux=False,
    )


def energy_invariant_identity_check(ref: dict[str, Any]) -> dict[str, tuple[float, float]]:
    field = "F_rad_energy_invariant_erg_per_cm2_s" if is_schema_v2(ref) else "F_rad_erg_per_cm2_s"
    return diffusion_identity_check(ref, flux_field=field, branch_oriented_flux=True)


def branch_identity_passed(stats: dict[str, tuple[float, float]]) -> bool:
    for branch in ("upstream", "downstream"):
        median_rel, p95_rel = stats[branch]
        if median_rel >= IDENTITY_MEDIAN_GATE or p95_rel >= IDENTITY_P95_GATE:
            return False
    return True


def print_identity_block(label: str, stats: dict[str, tuple[float, float]]) -> None:
    print(label)
    for branch in ("upstream", "downstream", "total"):
        median_rel, p95_rel = stats[branch]
        print(f"  {branch}: median_rel={median_rel:.3e} p95_rel={p95_rel:.3e}")


def exact_solution_field(sol: Any, *names: str) -> np.ndarray:
    for name in names:
        if hasattr(sol, name):
            return np.asarray(getattr(sol, name), dtype=float)
        try:
            return np.asarray(sol[name], dtype=float)
        except (KeyError, TypeError, ValueError, IndexError):
            pass
    dtype_names = getattr(getattr(sol, "dtype", None), "names", None)
    if dtype_names:
        for name in names:
            if name in dtype_names:
                return np.asarray(sol[name], dtype=float)
    raise KeyError(f"ExactPack solution missing field from {names}")


def transition_width(coord: np.ndarray, values: np.ndarray) -> float:
    order = np.argsort(coord)
    x = coord[order]
    y = values[order]
    y0 = float(np.nanmin(y))
    y1 = float(np.nanmax(y))
    if not y1 > y0:
        return math.inf
    lo = y0 + 0.10 * (y1 - y0)
    hi = y0 + 0.90 * (y1 - y0)
    x_lo = float(x[int(np.nanargmin(np.abs(y - lo)))])
    x_hi = float(x[int(np.nanargmin(np.abs(y - hi)))])
    return abs(x_hi - x_lo)


def rel_linf(got: np.ndarray, ref: np.ndarray) -> float:
    mask = np.isfinite(got) & np.isfinite(ref)
    if np.count_nonzero(mask) == 0:
        return math.inf
    return float(np.max(np.abs(got[mask] - ref[mask]) / np.maximum(np.abs(ref[mask]), TINY)))


def rel_l2(got: np.ndarray, ref: np.ndarray) -> float:
    mask = np.isfinite(got) & np.isfinite(ref)
    if np.count_nonzero(mask) == 0:
        return math.inf
    denom = float(np.sqrt(np.sum(ref[mask] * ref[mask])))
    return float(np.sqrt(np.sum((got[mask] - ref[mask]) ** 2)) / max(denom, TINY))


def build_exactpack_problem(ref: dict[str, Any], mach: float) -> tuple[Any | None, str]:
    if not EXACTPACK_AVAILABLE or radshocks is None:
        return None, "SKIPPED: ExactPack not installed"

    params = ref["parameters"]
    cv = (
        (1.0 + float(params["zbar"]))
        * float(params["eV_to_erg"])
        / (
            (float(params["gamma"]) - 1.0)
            * float(params["A_amu"])
            * float(params["proton_mass_g"])
        )
    )
    sigA = float(params["kappa_R_cm2_per_g"]) * float(params["rho0_g_per_cc"])
    kwargs = {
        "M0": mach,
        "rho0": float(params["rho0_g_per_cc"]),
        "Tref": float(params["T0_eV"]),
        "Cv": cv,
        "gamma": float(params["gamma"]),
        "sigA": sigA,
        "sigS": 0.0,
        "expDensity_abs": 1.0,
        "expTemp_abs": 0.0,
        "expDensity_scat": 1.0,
        "expTemp_scat": 0.0,
    }
    for class_name in ("ED_Solver", "ED_RadShock", "greyED_RadShock"):
        try:
            cls = getattr(radshocks, class_name)
        except AttributeError:
            continue
        try:
            return cls(**kwargs), f"Using ExactPack class: {class_name}"
        except TypeError:
            continue
    return None, "SKIPPED: could not find compatible ExactPack ED radshock class"


def compare_exactpack(ref: dict[str, Any], mach: float) -> int:
    problem, message = build_exactpack_problem(ref, mach)
    print(message)
    if problem is None:
        return 0

    table = ref["table"]
    x_ref = table_field(table, "x_cm")
    try:
        sol = problem(x_ref, 0.0)
    except TypeError:
        sol = problem(x_ref)
    exact = {
        "rho": exact_solution_field(sol, "rho", "density", "Density"),
        "T": exact_solution_field(sol, "T", "temperature", "Tm"),
        "u": exact_solution_field(sol, "u", "velocity", "Speed"),
    }
    in_house = {
        "rho": table_field(table, "rho_g_per_cc"),
        "T": table_field(table, "T_eV"),
        "u": table_field(table, "u_cm_per_s"),
    }

    rel_inf: dict[str, float] = {}
    rel_l2s: dict[str, float] = {}
    for field in ("rho", "T", "u"):
        candidates = [exact[field]]
        if field == "u":
            candidates.append(-exact[field])
        best = min(
            ((rel_l2(candidate, in_house[field]), candidate) for candidate in candidates),
            key=lambda item: item[0],
        )[1]
        rel_inf[field] = rel_linf(best, in_house[field])
        rel_l2s[field] = rel_l2(best, in_house[field])

    width_in_house = transition_width(x_ref, in_house["T"])
    width_exact = transition_width(x_ref, exact["T"])
    width_mismatch = abs(width_in_house - width_exact) / max(abs(width_exact), TINY)

    pass_inf = 1.0e-2
    pass_l2 = 5.0e-3
    pass_width = 2.0e-2
    failed: list[str] = []
    for field in ("rho", "T", "u"):
        if rel_inf[field] > pass_inf:
            failed.append(f"{field} L_inf={rel_inf[field]:.3e} > {pass_inf:.3e}")
        if rel_l2s[field] > pass_l2:
            failed.append(f"{field} L2={rel_l2s[field]:.3e} > {pass_l2:.3e}")
    if width_mismatch > pass_width:
        failed.append(f"width={width_mismatch:.3e} > {pass_width:.3e}")

    print(
        "ExactPack comparison: "
        + " ".join(
            f"{field}_Linf={rel_inf[field]:.3e} {field}_L2={rel_l2s[field]:.3e}"
            for field in ("rho", "T", "u")
        )
        + f" width_mismatch={width_mismatch:.3e}"
    )
    if failed:
        print("FAIL: ExactPack vs in-house disagreement:")
        for line in failed:
            print(f"  {line}")
        return 2
    print("PASS: ExactPack vs in-house agreement within thresholds")
    return 0


def reference_paths(reference: Path | None, mach: float) -> list[Path]:
    ref_path = reference or default_reference_path(mach)
    if ref_path.is_dir():
        paths = sorted(path for path in ref_path.glob("lr07_eda_M*.json") if path.is_file())
        if not paths:
            raise FileNotFoundError(f"no lr07_eda_M*.json references under {ref_path}")
        return paths
    return [ref_path]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=float, default=2.0)
    parser.add_argument("--reference", type=Path, default=None)
    args = parser.parse_args()

    paths = reference_paths(args.reference, args.M)
    identity_failed = False
    exactpack_status = 0
    exactpack_enabled = len(paths) == 1
    for ref_path in paths:
        ref = read_reference(ref_path)
        mach = float(ref.get("parameters", {}).get("M0", args.M))
        print(f"{ref_path} (M={mach:g})")
        if not is_schema_v2(ref):
            print(
                "WARNING: legacy LR07 schema detected; interpreting F_rad_erg_per_cm2_s "
                "as the schema-v1 branch-oriented flux fallback."
            )
        identity = in_house_diffusion_identity_check(ref)
        print_identity_block("In-house reference diffusion identity (monotone x_cm flux):", identity)
        energy_identity = energy_invariant_identity_check(ref)
        print_identity_block(
            "Energy-invariant branch-oriented flux sanity check:",
            energy_identity,
        )
        if not branch_identity_passed(identity):
            identity_failed = True
            print(
                "FAIL: branch-wise monotone-coordinate diffusion identity exceeds "
                f"median<{IDENTITY_MEDIAN_GATE:.1e}, p95<{IDENTITY_P95_GATE:.1e}"
            )
        elif exactpack_enabled:
            exactpack_status = max(exactpack_status, compare_exactpack(ref, mach))

    if identity_failed:
        print("This indicates a unit/dimensional issue in the LR07 generator, NOT in TENRYU.")
        return 1
    if not exactpack_enabled:
        print("SKIPPED: ExactPack comparison is only run for a single reference file")
    return exactpack_status


if __name__ == "__main__":
    raise SystemExit(main())
