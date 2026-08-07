#!/usr/bin/env python3
"""Cross-check frozen constant-Eddington FLD nED schema-v3 references."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


TINY = 1.0e-300
IDENTITY_MEDIAN_GATE = 1.0e-3
IDENTITY_P95_GATE = 1.0e-2
LOW_PRAD_GATE = 1.0e-3
REFERENCE_MODEL = "FLD_const_Eddington_nED_low_Prad"
PROBLEM = "FLD_const_Eddington"


def safe_float_token(value: float) -> str:
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def default_reference_path(mach: float) -> Path:
    repo = Path(__file__).resolve().parents[2]
    return repo / "tests" / "verification" / "data" / "fld_const_eddington_reference" / (
        f"fld_ced_M{safe_float_token(mach)}.json"
    )


def read_reference(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def is_schema_v3(ref: dict[str, Any]) -> bool:
    version = ref.get("schema_version")
    return version == 3 or str(version).strip() == "3" or str(version).strip().endswith(".v3")


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
    e_rad: np.ndarray,
    f_ref: np.ndarray,
    branches: np.ndarray,
    branch: str,
    c_light: float,
    kappa_R: float,
) -> np.ndarray:
    mask = branches == branch
    if np.count_nonzero(mask) < 3:
        return np.asarray([math.inf], dtype=float)
    order = np.argsort(x[mask])
    x_b = x[mask][order]
    rho_b = rho[mask][order]
    e_b = e_rad[mask][order]
    f_b = f_ref[mask][order]
    f_grad = -c_light / (3.0 * kappa_R * np.maximum(rho_b, TINY)) * np.gradient(e_b, x_b)
    rel = np.abs((f_b - f_grad) / np.maximum(np.abs(f_b) + np.abs(f_grad), TINY))
    return rel[10:-10] if rel.size > 20 else rel


def summarize_identity_rels(rels: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for branch, values in rels.items():
        out[branch] = (float(np.median(values)), float(np.percentile(values, 95.0)))
    total_values = np.concatenate([rels["upstream"], rels["downstream"]])
    out["total"] = (float(np.median(total_values)), float(np.percentile(total_values, 95.0)))
    return out


def diffusion_identity_check(ref: dict[str, Any]) -> dict[str, tuple[float, float]]:
    table = ref["table"]
    params = ref["parameters"]
    x = table_field(table, "x_cm")
    rho = table_field(table, "rho_g_per_cc")
    e_rad = table_field(table, "E_rad_erg_per_cm3")
    f_ref = table_field(table, "F_rad_erg_per_cm2_s")
    branches = branch_labels(table, x)
    c_light = float(params["c_cm_per_s"])
    kappa_R = float(params["kappa_R_cm2_per_g"])
    rels = {
        branch: branch_identity_rel(x, rho, e_rad, f_ref, branches, branch, c_light, kappa_R)
        for branch in ("upstream", "downstream")
    }
    return summarize_identity_rels(rels)


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


def check_schema_and_provenance(ref: dict[str, Any], path: Path) -> list[str]:
    failures: list[str] = []
    if not is_schema_v3(ref):
        failures.append(f"{path}: schema_version is not 3")
    if ref.get("problem") != PROBLEM:
        failures.append(f"{path}: problem={ref.get('problem')!r}, expected {PROBLEM!r}")
    if ref.get("reference_model") != REFERENCE_MODEL:
        failures.append(
            f"{path}: reference_model={ref.get('reference_model')!r}, expected {REFERENCE_MODEL!r}"
        )
    provenance = ref.get("provenance", {})
    if provenance.get("reference_model") != REFERENCE_MODEL:
        failures.append(f"{path}: provenance reference_model does not match top-level reference_model")
    table = ref.get("table", {})
    for field in (
        "x_cm",
        "rho_g_per_cc",
        "u_cm_per_s",
        "T_eV",
        "T_rad_eV",
        "E_rad_erg_per_cm3",
        "P_mat_dyne_per_cm2",
        "P_rad_dyne_per_cm2",
        "F_rad_erg_per_cm2_s",
        "F_rad_energy_invariant_erg_per_cm2_s",
        "F_rad_lab_erg_per_cm2_s",
        "branch",
    ):
        if field not in table:
            failures.append(f"{path}: missing table field {field}")
    for state_name in ("upstream", "downstream", "shock_upstream", "shock_downstream"):
        if state_name not in ref.get("states", {}):
            failures.append(f"{path}: missing states.{state_name}")
    return failures


def low_prad_passed(ref: dict[str, Any]) -> tuple[bool, float]:
    prad = float(ref.get("diagnostics", {}).get("max_P_rad_over_P_mat", math.inf))
    return prad <= LOW_PRAD_GATE, prad


def conservation_passed(ref: dict[str, Any]) -> tuple[bool, float]:
    residual = float(ref.get("diagnostics", {}).get("conservation_residual_rel_max", math.inf))
    return residual <= 1.0e-12, residual


def shock_match_passed(ref: dict[str, Any]) -> tuple[bool, float, float]:
    up = ref.get("states", {}).get("shock_upstream", {})
    dn = ref.get("states", {}).get("shock_downstream", {})
    e_up = float(up.get("E_rad", math.nan))
    e_dn = float(dn.get("E_rad", math.nan))
    f_up = float(up.get("F_rad", math.nan))
    f_dn = float(dn.get("F_rad", math.nan))
    e_rel = abs(e_up - e_dn) / max(abs(e_up), abs(e_dn), TINY)
    f_rel = abs(f_up - f_dn) / max(abs(f_up), abs(f_dn), TINY)
    return e_rel <= 1.0e-12 and f_rel <= 1.0e-12, e_rel, f_rel


def reference_paths(reference: Path | None, mach: float) -> list[Path]:
    ref_path = reference or default_reference_path(mach)
    if ref_path.is_dir():
        paths = sorted(path for path in ref_path.glob("fld_ced_M*.json") if path.is_file())
        if not paths:
            raise FileNotFoundError(f"no fld_ced_M*.json references under {ref_path}")
        return paths
    return [ref_path]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=float, default=2.0)
    parser.add_argument("--reference", type=Path, default=None)
    args = parser.parse_args()

    paths = reference_paths(args.reference, args.M)
    failed = False
    for ref_path in paths:
        ref = read_reference(ref_path)
        mach = float(ref.get("parameters", {}).get("M0", args.M))
        print(f"{ref_path} (M={mach:g})")
        schema_failures = check_schema_and_provenance(ref, ref_path)
        for failure in schema_failures:
            print(f"FAIL: {failure}")
        failed = failed or bool(schema_failures)

        identity = diffusion_identity_check(ref)
        print_identity_block("FLD-CED reference diffusion identity:", identity)
        if not branch_identity_passed(identity):
            failed = True
            print(
                "FAIL: branch-wise diffusion identity exceeds "
                f"median<{IDENTITY_MEDIAN_GATE:.1e}, p95<{IDENTITY_P95_GATE:.1e}"
            )

        low_prad_ok, prad = low_prad_passed(ref)
        print(f"Low-P_rad diagnostics: max_P_rad/P_mat={prad:.3e}")
        if not low_prad_ok:
            failed = True
            print(f"FAIL: low-P_rad gate exceeds {LOW_PRAD_GATE:.1e}")

        conservation_ok, conservation = conservation_passed(ref)
        print(f"Conservation diagnostics: residual={conservation:.3e}")
        if not conservation_ok:
            failed = True
            print("FAIL: reference conservation exceeds 1.0e-12")

        shock_ok, e_rel, f_rel = shock_match_passed(ref)
        print(f"Shock match: E_rad_rel={e_rel:.3e} F_rad_rel={f_rel:.3e}")
        if not shock_ok:
            failed = True
            print("FAIL: shock E_rad/F_rad continuity exceeds 1.0e-12")

    if failed:
        print("FLD-CED reference check FAILED")
        return 1
    print("PASS: all FLD-CED schema-v3 branch identity checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
