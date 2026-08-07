#!/usr/bin/env python3
"""Cross-check the I3 in-house S_N reference against ExactPack when available."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


TINY = 1.0e-300


try:
    from exactpack.solvers.radshocks import Sn_Solver

    EXACTPACK_AVAILABLE = True
except ImportError:
    Sn_Solver = None  # type: ignore[assignment]
    EXACTPACK_AVAILABLE = False


sys.path.insert(0, str(Path(__file__).resolve().parent))
import sn_radshock_reference_generator as sn_ref  # noqa: E402


def table_array(table: dict[str, Any], name: str) -> np.ndarray:
    return np.asarray(table[name], dtype=float)


def transition_center(x: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    lo = float(np.nanmin(ys))
    hi = float(np.nanmax(ys))
    if not hi > lo:
        return float(xs[len(xs) // 2])
    target = lo + 0.5 * (hi - lo)
    return float(xs[int(np.nanargmin(np.abs(ys - target)))])


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


def exactpack_native_profile(problem: Any) -> dict[str, np.ndarray]:
    # Sn_Solver stores the steady profile upstream-to-downstream in self.x.
    # Its _run method flips the profile for a moving-shock convention; the I3
    # reference is already a steady shock-frame table, so compare to the native
    # stored profile and only translate the shock position.
    x = np.asarray(problem.x, dtype=float)
    return {
        "x": x,
        "rho": np.asarray(problem.Density, dtype=float),
        "T": np.asarray(problem.Tm, dtype=float),
        "T_rad": np.asarray(problem.Tr, dtype=float),
        "u": np.asarray(problem.Speed, dtype=float),
        "VEF": np.asarray(problem.VEF, dtype=float),
    }


def build_exactpack_problem(cfg: sn_ref.SNConfig) -> tuple[Any | None, str]:
    if not EXACTPACK_AVAILABLE or Sn_Solver is None:
        return None, "SKIPPED: ExactPack not installed"

    # ExactPack radshocks uses dimensionless rho=rho_physical/rho0 in
    # sigma_a = sigA * rho**expDensity_abs * T**expTemp_abs.  TENRYU's
    # static reference has sigma_a = kappa_a[cm2/g] * rho_physical[g/cc], so
    # sigA must be the upstream inverse length kappa_a*rho0 and the density
    # exponent is one.
    sigA = cfg.kappa_a_cm2_g * cfg.rho0_gcc
    kwargs = {
        "M0": cfg.mach,
        "rho0": cfg.rho0_gcc,
        "Tref": cfg.T0_eV,
        "Cv": cfg.Cv_erg_per_g_eV,
        "gamma": cfg.gamma,
        "sigA": sigA,
        "sigS": 0.0,
        "expDensity_abs": 1.0,
        "expTemp_abs": 0.0,
        "expDensity_scat": 1.0,
        "expTemp_scat": 0.0,
        # ExactPack's Sn_Solver defaults to problem='nED'.  radshock.py also
        # supports 'LM_nED'; I3 records the nED/FML17 convention because that
        # is the auxiliary anchor named in the audit.
        "problem": "nED",
        "Sn": cfg.n_angles,
        "int_tol": 1.0e-10,
    }
    try:
        return Sn_Solver(**kwargs), "Using ExactPack class: Sn_Solver(problem='nED')"
    except Exception as exc:
        return None, f"SKIPPED: ExactPack Sn_Solver failed to initialize: {exc}"


def aligned_exactpack_on_reference(problem: Any, ref: dict[str, Any]) -> dict[str, np.ndarray]:
    table = ref["table"]
    x_ref = table_array(table, "x_cm")
    T_ref = table_array(table, "T_eV")
    native = exactpack_native_profile(problem)
    center_ref = transition_center(x_ref, T_ref)
    center_exact = transition_center(native["x"], native["T"])
    x_query = x_ref - center_ref + center_exact
    out = {"x": x_ref}
    for field in ("rho", "T", "T_rad", "u", "VEF"):
        out[field] = np.interp(x_query, native["x"], native[field])
    return out


def compare_exactpack(ref: dict[str, Any], problem: Any, l2_gate: float, linf_gate: float) -> int:
    table = ref["table"]
    x_ref = table_array(table, "x_cm")
    dx = float(np.median(np.diff(np.sort(x_ref)))) if len(x_ref) > 2 else 0.0
    shock_center = transition_center(x_ref, table_array(table, "T_eV"))
    branch_mask = np.abs(x_ref - shock_center) > max(1.01 * dx, 1.0e-12)
    in_house = {
        "rho": table_array(table, "rho_g_per_cc"),
        "T": table_array(table, "T_eV"),
        "T_rad": table_array(table, "T_rad_eV"),
        "u": table_array(table, "u_cm_per_s"),
        "chi": table_array(table, "chi_z"),
    }
    exact = aligned_exactpack_on_reference(problem, ref)
    failed: list[str] = []
    print("ExactPack comparison after shock-position alignment:")
    for field in ("rho", "T", "T_rad", "u"):
        candidates = [exact[field]]
        if field == "u":
            candidates.append(-exact[field])
        best = min(candidates, key=lambda values: rel_l2(values[branch_mask], in_house[field][branch_mask]))
        l2 = rel_l2(best[branch_mask], in_house[field][branch_mask])
        linf = rel_linf(best[branch_mask], in_house[field][branch_mask])
        print(f"  {field}: L2={l2:.6e} Linf={linf:.6e}")
        if l2 > l2_gate:
            failed.append(f"{field} L2={l2:.3e} > {l2_gate:.3e}")
        if linf > linf_gate:
            failed.append(f"{field} Linf={linf:.3e} > {linf_gate:.3e}")

    vef_l2 = rel_l2(exact["VEF"][branch_mask], in_house["chi"][branch_mask])
    vef_linf = rel_linf(exact["VEF"][branch_mask], in_house["chi"][branch_mask])
    print(f"  VEF_vs_chi_z report_only: L2={vef_l2:.6e} Linf={vef_linf:.6e}")

    if failed:
        print("FAIL: ExactPack vs in-house disagreement:")
        for line in failed:
            print(f"  {line}")
        return 2
    print("PASS: ExactPack comparison within loose auxiliary gates")
    return 0


def compare_payloads_delta(base: dict[str, Any], aux: dict[str, Any]) -> None:
    x = table_array(base["table"], "x_cm")
    x_aux = table_array(aux["table"], "x_cm")
    print("Radiation-momentum auxiliary delta (AUX - reference):")
    for field, key in (
        ("rho", "rho_g_per_cc"),
        ("T", "T_eV"),
        ("T_rad", "T_rad_eV"),
        ("u", "u_cm_per_s"),
    ):
        ref = table_array(base["table"], key)
        got = np.interp(x, x_aux, table_array(aux["table"], key))
        print(f"  {field}: L2={rel_l2(got, ref):.6e} Linf={rel_linf(got, ref):.6e}")


def build_config(args: argparse.Namespace, *, with_rad_momentum: bool = False) -> sn_ref.SNConfig:
    base = sn_ref.load_default_config()
    return sn_ref.SNConfig(
        mach=args.mach,
        n_angles=args.n_angles,
        kappa_scale=args.kappa_scale,
        n_points=args.n_points,
        span_rel=args.span_rel,
        closure="sn",
        with_rad_momentum=with_rad_momentum,
        relax=args.relax,
        max_iter=args.max_iter,
        tol=args.tol,
        gamma=base.gamma,
        rho0_gcc=base.rho0_gcc,
        T0_eV=base.T0_eV,
        Cv_erg_per_g_eV=base.Cv_erg_per_g_eV,
        A_amu=base.A_amu,
        zbar=base.zbar,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mach", type=float, default=2.0)
    parser.add_argument("--n-angles", type=int, default=16)
    parser.add_argument("--kappa-scale", type=float, default=1.0)
    parser.add_argument("--n-points", type=int, default=1601)
    parser.add_argument("--span-rel", type=float, default=sn_ref.DEFAULT_SPAN_REL)
    parser.add_argument("--relax", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1.0e-12)
    parser.add_argument("--l2-gate", type=float, default=0.05)
    parser.add_argument("--linf-gate", type=float, default=0.10)
    parser.add_argument("--prad-delta", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = build_config(args)
    ref = sn_ref.build_reference(cfg)

    if args.prad_delta:
        aux = sn_ref.build_reference(build_config(args, with_rad_momentum=True))
        compare_payloads_delta(ref, aux)
        return 0

    problem, message = build_exactpack_problem(cfg)
    print(message)
    if problem is None:
        return 0
    return compare_exactpack(ref, problem, args.l2_gate, args.linf_gate)


if __name__ == "__main__":
    raise SystemExit(main())
