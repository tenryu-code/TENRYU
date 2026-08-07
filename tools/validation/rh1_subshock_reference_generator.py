#!/usr/bin/env python3
"""Generate the RH1 energy-only radiative-shock subshock reference.

This is a thin RH1 wrapper around the frozen 1T FLD-CED generator.  The RH1
upstream state has a much lower radiation-pressure fraction than the frozen I1
tables, so the wrapper widens only the shock-flux bracket scan; all ODEs,
Hugoniot matching, table construction, and diagnostics are imported from the
1T generator.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fld_const_eddington_reference_generator as one_t


BRACKET_FRACTION_MIN = 1.0e-8
BRACKET_FRACTION_MAX = 0.75
BRACKET_SAMPLES = 96
SCHEMA_VERSION = "tenryu.rh1_subshock_reference.v1"
REFERENCE_MODEL = "RH1_energy_only_FLD_CED_subshock"

_LAST_BRACKET_DIAGNOSTICS: dict[str, Any] = {}


def _strip_generated_utc(payload: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(payload, sort_keys=True))
    copied.get("provenance", {}).pop("generated_utc", None)
    return copied


def solve_shock_match_rh1(cfg: Any, inv: Any) -> Any:
    """1T shock match with a wider low-flux bracket for RH1 parameters."""
    global _LAST_BRACKET_DIAGNOSTICS
    F_min_up = one_t.radiation_flux_energy(
        inv.T_sonic * (1.0 - 1.0e-8), "upstream", inv
    )
    F_min_dn = one_t.radiation_flux_energy(
        inv.T_sonic * (1.0 - 1.0e-8), "downstream", inv
    )
    F_min_common = max(F_min_up, F_min_dn)
    if not F_min_common < 0.0:
        raise RuntimeError("common shock-flux bracket is not negative")

    samples = [
        F_min_common * frac
        for frac in np.geomspace(
            BRACKET_FRACTION_MIN, BRACKET_FRACTION_MAX, BRACKET_SAMPLES
        )
    ]
    bracket: tuple[float, float] | None = None
    previous: tuple[float, float] | None = None
    evaluated = 0
    for F_value in samples:
        try:
            mismatch = one_t.shock_mismatch(cfg, inv, float(F_value))
        except (RuntimeError, FloatingPointError, ValueError):
            previous = None
            continue
        evaluated += 1
        if previous is not None and previous[1] * mismatch < 0.0:
            bracket = (previous[0], float(F_value))
            break
        previous = (float(F_value), mismatch)
    if bracket is None:
        raise RuntimeError("failed to bracket continuous E_rad shock match")

    F_shock = float(
        brentq(
            lambda F: one_t.shock_mismatch(cfg, inv, F),
            bracket[0],
            bracket[1],
            xtol=max(abs(F_min_common) * cfg.shooting_rtol, 1.0e6),
            rtol=cfg.shooting_rtol,
            maxiter=100,
        )
    )
    up = one_t.integrate_branch(cfg, inv, F_shock, "upstream")
    dn = one_t.integrate_branch(cfg, inv, F_shock, "downstream")
    E_shock = 0.5 * (float(up["E_shock"]) + float(dn["E_shock"]))
    _LAST_BRACKET_DIAGNOSTICS = {
        "bracket_fraction_min": BRACKET_FRACTION_MIN,
        "bracket_fraction_max": BRACKET_FRACTION_MAX,
        "bracket_samples": BRACKET_SAMPLES,
        "evaluated_samples_before_bracket": evaluated,
        "F_min_common_erg_per_cm2_s": F_min_common,
        "F_shock_fraction_of_F_min_common": F_shock / F_min_common,
        "F_bracket_erg_per_cm2_s": [bracket[0], bracket[1]],
        "reason": (
            "RH1 T0=1 eV has a shock-flux root below the frozen I1 "
            "solve_shock_match scan lower fraction of 1e-3."
        ),
    }
    return one_t.ShockMatch(
        F_shock=F_shock,
        T_upstream_shock=float(up["T_shock"]),
        T_downstream_shock=float(dn["T_shock"]),
        E_shock=E_shock,
        T_rad_shock=(E_shock / one_t.A_EV) ** 0.25,
        upstream_length_cm=float(up["length_cm"]),
        downstream_length_cm=float(dn["length_cm"]),
    )


def _add_consumer_aliases(payload: dict[str, Any]) -> None:
    table = payload["table"]
    payload["schema_version_rh1"] = SCHEMA_VERSION
    payload["reference_model_rh1"] = REFERENCE_MODEL
    payload["x_cm"] = table["x_cm"]
    payload["rho_g_per_cc"] = table["rho_g_per_cc"]
    payload["u_cm_per_s"] = table["u_cm_per_s"]
    payload["T_mat_eV"] = table["T_eV"]
    payload["T_rad_eV"] = table["T_rad_eV"]
    payload["E_r_erg_per_cc"] = table["E_rad_erg_per_cm3"]
    payload["F_r_erg_per_cm2_s"] = table["F_rad_erg_per_cm2_s"]
    payload["rh1_consumer_aliases"] = {
        "T_mat_eV": "table.T_eV",
        "E_r_erg_per_cc": "table.E_rad_erg_per_cm3",
        "F_r_erg_per_cm2_s": "table.F_rad_erg_per_cm2_s",
        "x_cm": "table.x_cm",
        "rho_g_per_cc": "table.rho_g_per_cc",
        "u_cm_per_s": "table.u_cm_per_s",
        "T_rad_eV": "table.T_rad_eV",
    }


def build_reference(cfg: Any, cli_args: dict[str, Any]) -> dict[str, Any]:
    original_solver = one_t.solve_shock_match
    one_t.solve_shock_match = solve_shock_match_rh1
    try:
        payload_a = one_t.build_reference(cfg)
        bracket_a = dict(_LAST_BRACKET_DIAGNOSTICS)
        payload_b = one_t.build_reference(cfg)
    finally:
        one_t.solve_shock_match = original_solver

    if _strip_generated_utc(payload_a) != _strip_generated_utc(payload_b):
        raise RuntimeError("RH1 generator self-check failed: live reruns differ")

    payload_a["provenance"]["generator"] = (
        "tools/validation/rh1_subshock_reference_generator.py"
    )
    payload_a["provenance"]["base_generator"] = (
        "tools/validation/fld_const_eddington_reference_generator.py"
    )
    payload_a["provenance"]["rh1_bracket_extension"] = bracket_a
    payload_a["provenance"]["cli_args"] = cli_args
    payload_a["provenance"]["sv1_live_rerun_deep_equal_after_timestamp_strip"] = True
    _add_consumer_aliases(payload_a)
    return payload_a


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="tests/verification/data/rh1_subshock_reference",
    )
    parser.add_argument("--mach", type=float, default=3.0)
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--rho0-gcc", type=float, default=1.0)
    parser.add_argument("--T0-eV", type=float, default=1.0)
    parser.add_argument("--kappa-R-cm2-g", type=float, default=1.0)
    parser.add_argument("--A-amu", type=float, default=1.0)
    parser.add_argument("--zbar", type=float, default=1.0)
    parser.add_argument("--n-points", type=int, default=1601)
    parser.add_argument("--target-domain-span-cm", type=float, default=10.0)
    parser.add_argument("--padding-points-per-side", type=int, default=8)
    parser.add_argument("--ode-rtol", type=float, default=1.0e-9)
    parser.add_argument("--shooting-rtol", type=float, default=1.0e-10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = one_t.FLDCEDConfig(
        mach=args.mach,
        gamma=args.gamma,
        rho0_gcc=args.rho0_gcc,
        T0_eV=args.T0_eV,
        kappa_R_cm2_g=args.kappa_R_cm2_g,
        A_amu=args.A_amu,
        zbar=args.zbar,
        n_points=args.n_points,
        target_domain_span_cm=args.target_domain_span_cm,
        padding_points_per_side=args.padding_points_per_side,
        ode_rtol=args.ode_rtol,
        shooting_rtol=args.shooting_rtol,
    )
    payload = build_reference(cfg, vars(args).copy())
    out_dir = Path(args.out_dir)
    token = one_t.safe_float_token(args.mach)
    filename = f"rh1_subshock_M{token}.json"
    out_path = out_dir / filename
    one_t.write_json(out_path, payload)

    diag = payload["diagnostics"]
    x = payload["table"]["x_cm"]
    manifest = {
        "schema_version": "tenryu.rh1_subshock_reference_manifest.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "tools/validation/rh1_subshock_reference_generator.py",
        "base_generator": "tools/validation/fld_const_eddington_reference_generator.py",
        "git_commit": one_t.git_commit(),
        "rows": [
            {
                "file": filename,
                "M0": payload["parameters"]["M0"],
                "gamma": payload["parameters"]["gamma"],
                "rho0_g_per_cc": payload["parameters"]["rho0_g_per_cc"],
                "T0_eV": payload["parameters"]["T0_eV"],
                "kappa_R_cm2_per_g": payload["parameters"]["kappa_R_cm2_per_g"],
                "A_amu": payload["parameters"]["A_amu"],
                "zbar": payload["parameters"]["zbar"],
                "n_points": len(x),
                "x_min_cm": float(x[0]),
                "x_max_cm": float(x[-1]),
                "domain_span_cm": float(x[-1] - x[0]),
                "max_P_rad_over_P_mat": diag["max_P_rad_over_P_mat"],
                "conservation_residual_rel_max": diag[
                    "conservation_residual_rel_max"
                ],
                "reference_identity_upstream_p95_rel": diag[
                    "reference_identity_upstream_p95_rel"
                ],
                "reference_identity_downstream_p95_rel": diag[
                    "reference_identity_downstream_p95_rel"
                ],
                "sv1_live_rerun": True,
                "F_shock_fraction_of_F_min_common": payload["provenance"][
                    "rh1_bracket_extension"
                ]["F_shock_fraction_of_F_min_common"],
            }
        ],
    }
    one_t.write_json(out_dir / "manifest.json", manifest)
    print(
        f"{out_path}: SV1=PASS conservation={diag['conservation_residual_rel_max']:.3e} "
        f"identity_up_p95={diag['reference_identity_upstream_p95_rel']:.3e} "
        f"identity_dn_p95={diag['reference_identity_downstream_p95_rel']:.3e} "
        f"max_Prad_over_Pmat={diag['max_P_rad_over_P_mat']:.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
