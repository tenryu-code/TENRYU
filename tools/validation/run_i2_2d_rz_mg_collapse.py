#!/usr/bin/env python3
"""I2 2D RZ multigroup FLD grey-collapse battery harness.

Runs the i2_mgfld_collapse_2d_rz_slab.py deck over the adjudicated run
matrix (spec v3.1 §7) and writes a summary.json consumed by the
[.expensive] Catch2 wrapper:
  coupled: R0 grey + R2 mg {2,4,8} flat  -> collapse gates vs R0
           (eps_L2<=1e-4, eps_Linf<=1e-3 on radial-mean z-profiles),
           flat N_g-ladder pairwise consistency, hygiene gates.
  front:   R3 mg {2,4,8} structured + R3g grey comparator pair at
           kappa_P/kappa_R(T_hot) -> departure metrics (REPORTED + Cauchy
           contraction gate), Richardson nz-ladder at N_g=4 (contraction
           reported; window frozen by main at first battery, Q1 ruling).

Absolute metrics vs the FLD-CED ODE reference are REPORTED only
(two_state-mode calibration caveat, adjudication Q6); the binding collapse
comparator is the same-binary R0 grey run. Reuses the I1-A harness readers
(run_i1_2d_rz_fld_ced.read_snapshot sums the HDF5 group axis into a grey
total via run_i1_le08.reshape_rad).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import run_i1_le08 as base  # noqa: E402
import run_i1_2d_rz_fld_ced as i1  # noqa: E402

LADDER_ROW = "i2_mgfld_collapse_2d_rz_slab"
TINY = 1.0e-300

# Binding gates (spec v3.1 §7; roadmap row 536 I2b tier).
COUPLED_EPS_L2_GATE = 1.0e-4
COUPLED_EPS_LINF_GATE = 1.0e-3
LADDER_PAIRWISE_L2_GATE = 1.0e-4
RADIAL_INVARIANCE_GATE = 1.0e-6
FRONT_CAUCHY_CONTRACTION_GATE = 0.75

# freq_dep_marshak analytic law (src/materials/opacity.cu:103-110):
#   sigma_nu = (27e9 / nu_eV^3) * (1 - exp(-nu/T))  [1/cm at the model's
#   fixed normalization]; kappa = sigma/rho.
SIGMA0_EV3_PER_CM = 27.0e9
BAND_LO_EV = 1.0
BAND_HI_EV = 1.5e3


def planck_weight(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    small = x < 700.0
    ex = np.expm1(x[small])
    good = ex > 0.0
    vals = np.zeros_like(x[small])
    vals[good] = x[small][good] ** 3 / ex[good]
    out[small] = vals
    big = ~small
    out[big] = x[big] ** 3 * np.exp(-x[big])
    return out


def rosseland_weight(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    small = x < 80.0
    em1 = np.expm1(x[small])
    good = em1 > 0.0
    vals = np.zeros_like(x[small])
    ex = np.exp(x[small][good])
    vals[good] = x[small][good] ** 4 * ex / (em1[good] ** 2)
    out[small] = vals
    big = ~small
    out[big] = x[big] ** 4 * np.exp(-x[big])
    return out


def freq_dep_band_means(T_eV: float, rho_gcc: float) -> dict[str, float]:
    """Full-band [1,1500] eV Planck & Rosseland means of the freq_dep law.

    Planck mean collapses analytically: sigma*B ~ 27e9 * x^3 exp(-x)/x^3
    -> numerator integrand 27e9 * exp(-x); computed numerically anyway on
    the same grid for symmetry with the Rosseland mean.
    """
    x = np.logspace(
        math.log10(BAND_LO_EV / T_eV), math.log10(BAND_HI_EV / T_eV), 4096
    )
    nu = x * T_eV
    sigma = (SIGMA0_EV3_PER_CM / nu**3) * (-np.expm1(-x))
    b = planck_weight(x)
    r = rosseland_weight(x)
    sigma_p = float(np.trapz(sigma * b, x) / max(np.trapz(b, x), TINY))
    inv_sigma_r = float(np.trapz(r / np.maximum(sigma, TINY), x) / max(np.trapz(r, x), TINY))
    sigma_r = 1.0 / max(inv_sigma_r, TINY)
    return {
        "kappa_P_cm2g": sigma_p / max(rho_gcc, TINY),
        "kappa_R_cm2g": sigma_r / max(rho_gcc, TINY),
    }


def eps_norms(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    cand = np.asarray(candidate, dtype=float)
    ref = np.asarray(reference, dtype=float)
    diff = cand - ref
    l2 = float(np.linalg.norm(diff) / max(np.linalg.norm(ref), TINY))
    ref_scale = float(np.max(np.abs(ref)))
    linf = float(np.max(np.abs(diff)) / max(ref_scale, TINY))
    return {"eps_L2": l2, "eps_Linf": linf}


def radial_invariance_max_rel(field2d: np.ndarray) -> float:
    arr = np.asarray(field2d, dtype=float)
    col_mean = np.mean(arr, axis=0)
    scale = np.maximum(np.abs(col_mean), TINY)
    return float(np.max(np.abs(arr - col_mean[None, :]) / scale[None, :]))


def latest_snapshot(outdir: Path, nr: int, nz: int) -> Any:
    actual = base.discover_actual_outdir(outdir)
    results_dir = actual / "results"
    candidates = sorted(results_dir.glob("*.h5")) + sorted(actual.glob("*.h5"))
    if not candidates:
        raise FileNotFoundError(
            f"no HDF5 snapshots under {results_dir} or {actual}"
        )
    best = None
    for path in candidates:
        try:
            snap = i1.read_snapshot(path, nr, nz, "2D_RZ")
        except Exception:
            continue
        if best is None or snap.t > best.t:
            best = snap
    if best is None:
        raise RuntimeError(f"no readable snapshots under {actual}")
    return best


def run_deck(
    args: argparse.Namespace,
    label: str,
    env_extra: dict[str, str],
    nr: int,
    nz: int,
) -> dict[str, Any]:
    outdir = Path(args.out_root) / label
    outdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "TENRYU_I2_2D_RZ_NR": str(nr),
            "TENRYU_I2_2D_RZ_NZ": str(nz),
            "TENRYU_I2_2D_RZ_OUTDIR": str(outdir),
            "TENRYU_I2_2D_RZ_PRODUCTION_AUDIT_TIER": args.production_audit_tier,
            "TENRYU_I2_2D_RZ_VERBOSITY": args.verbosity,
        }
    )
    env.update(env_extra)
    log_path = outdir / "run.log"
    cmd = [str(args.tenryu_bin), "run", str(args.deck)]
    resumed = False
    if getattr(args, "resume", False) and log_path.exists():
        try:
            prior = i1.run_info(outdir)
        except Exception:
            prior = {}
        if str(prior.get("termination_reason", "")) == "t_end_reached":
            # Completed leg: reuse outputs; all downstream analysis
            # (snapshot re-read, newton/valve log parsing, hygiene and
            # effective-groups gates) runs unchanged on the existing files.
            resumed = True
            print(f"[run_i2_2d_rz_mg_collapse] resume: skipping {label}")
    if resumed:
        returncode = 0
    else:
        with log_path.open("w", encoding="utf-8") as log_handle:
            proc = subprocess.run(
                cmd, env=env, stdout=log_handle, stderr=subprocess.STDOUT, check=False
            )
        returncode = int(proc.returncode)
    info = {}
    try:
        info = i1.run_info(outdir)
    except Exception:
        info = {}
    result: dict[str, Any] = {
        "label": label,
        "returncode": returncode,
        "resumed": resumed,
        "outdir": str(outdir),
        "log": str(log_path),
        "termination_reason": str(info.get("termination_reason", "unknown")),
        "env": {k: v for k, v in env_extra.items()},
        "nr": nr,
        "nz": nz,
    }
    if returncode != 0:
        result["failed"] = True
        return result
    snap = latest_snapshot(outdir, nr, nz)
    result["t_final"] = float(snap.t)
    result["snapshot"] = snap
    n_cells = nr * nz
    try:
        newton = i1.parse_newton_diagnostic(log_path, n_cells)
        i1.repair_newton_residual_scale(newton, snap)
        result["newton"] = {
            k: v for k, v in newton.items() if isinstance(v, (int, float, bool, str))
        }
        result["newton_pass"] = bool(newton_complete_solve_gate(newton))
    except Exception as exc:  # pragma: no cover - diagnostic robustness
        result["newton"] = {"error": str(exc)}
        result["newton_pass"] = False
    try:
        valve_count = i1.escape_valve_jsonl_count(base.discover_actual_outdir(outdir))
        result["escape_valve_count"] = -1 if valve_count is None else int(valve_count)
        result["no_escape_valves"] = (valve_count is None) or (valve_count == 0)
    except Exception:
        result["escape_valve_count"] = -1
        result["no_escape_valves"] = True
    result["radial_invariance"] = {
        "rho": radial_invariance_max_rel(snap.rho),
        "Te": radial_invariance_max_rel(snap.Te),
        "Ti": radial_invariance_max_rel(snap.Ti),
        "E_rad": radial_invariance_max_rel(snap.rad_E),
    }
    result["radial_invariance_pass"] = all(
        v <= RADIAL_INVARIANCE_GATE for v in result["radial_invariance"].values()
    )
    return result


def profiles(snap: Any) -> dict[str, np.ndarray]:
    a_eV = 137.2
    e_total = np.asarray(base.profile_average(snap, snap.rad_E), dtype=float)
    out = {
        "rho": np.asarray(base.profile_average(snap, snap.rho), dtype=float),
        "Te": np.asarray(base.profile_average(snap, snap.Te), dtype=float),
        "Ti": np.asarray(base.profile_average(snap, snap.Ti), dtype=float),
        "E_rad": e_total,
        "T_rad_sigma": np.power(np.maximum(e_total, 0.0) / a_eV, 0.25),
        "u_z": np.asarray(i1.profile_velocity(snap), dtype=float),
    }
    return out


def z_centers(snap: Any) -> np.ndarray:
    zc = i1.cell_center_from_nodes(snap.z_nodes)
    return np.asarray(np.mean(zc, axis=0), dtype=float)


def wave12_report(snap: Any) -> dict[str, float]:
    prof = profiles(snap)
    d_tr = prof["T_rad_sigma"] / np.maximum(prof["Te"], TINY) - 1.0
    return {
        "max_dTr": float(np.max(np.abs(d_tr))),
        "max_T_rad_sigma": float(np.max(prof["T_rad_sigma"])),
        "max_Te": float(np.max(prof["Te"])),
    }


def front_metrics(snap: Any, t_hot: float, t_cold: float) -> dict[str, float]:
    te = profiles(snap)["Te"]
    z = z_centers(snap)
    norm = (te - t_cold) / max(t_hot - t_cold, TINY)

    def crossing(level: float) -> float:
        above = norm >= level
        if not bool(np.any(above)) or bool(np.all(above)):
            return float("nan")
        idx = int(np.max(np.nonzero(above)))
        if idx + 1 >= norm.size:
            return float(z[idx])
        y0, y1 = norm[idx], norm[idx + 1]
        if abs(y1 - y0) < 1.0e-30:
            return float(z[idx])
        w = (level - y0) / (y1 - y0)
        return float(z[idx] + w * (z[idx + 1] - z[idx]))

    z50 = crossing(0.5)
    z90 = crossing(0.9)
    z10 = crossing(0.1)
    width = abs(z10 - z90) if (math.isfinite(z10) and math.isfinite(z90)) else float("nan")
    return {"front_z_cm": z50, "front_width_cm": width}


def compare_profiles(mg: Any, ref: Any) -> dict[str, dict[str, float]]:
    mg_prof = profiles(mg)
    ref_prof = profiles(ref)
    return {name: eps_norms(mg_prof[name], ref_prof[name]) for name in mg_prof}


def coupled_env(groups: int) -> dict[str, str]:
    return {
        "TENRYU_I2_2D_RZ_GROUPS": str(groups),
        "TENRYU_I2_2D_RZ_OPACITY": "flat",
        "TENRYU_I2_2D_RZ_MODE": "coupled",
    }


def front_env(groups: int, opacity: str, kappa_override: float | None = None) -> dict[str, str]:
    env = {
        "TENRYU_I2_2D_RZ_GROUPS": str(groups),
        "TENRYU_I2_2D_RZ_OPACITY": opacity,
        "TENRYU_I2_2D_RZ_MODE": "front",
    }
    if kappa_override is not None:
        env["TENRYU_I2_2D_RZ_KAPPA_OVERRIDE_CM2G"] = f"{kappa_override:.17g}"
    return env


def assert_effective_groups(run: dict[str, Any], expected: int) -> bool:
    """Anti-tautology (spec F11): the deck banner must show the effective
    group count actually requested (auto-grey trap guard)."""
    try:
        text = Path(run["log"]).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    needle = f"[deck:i2_mgfld_collapse_2d_rz_slab:mg] groups={expected} "
    return needle in text


def newton_complete_solve_gate(newton: dict[str, Any]) -> bool:
    records = int(newton.get("records") or 0)
    cap_hit = int(newton.get("newton_cap_hit_count") or 0)
    invalid = int(newton.get("newton_invalid_count") or 0)
    return (
        records > 0
        and bool(newton.get("all_converged_count_is_complete_solves"))
        and invalid == 0
        and cap_hit == 0
    )


def hygiene_pass(run: dict[str, Any]) -> bool:
    return (
        run.get("returncode") == 0
        and run.get("termination_reason") == "t_end_reached"
        and bool(run.get("newton_pass"))
        and bool(run.get("no_escape_valves"))
        and bool(run.get("radial_invariance_pass"))
    )


def sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items() if k != "snapshot"}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenryu-bin", required=True)
    parser.add_argument("--deck", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--coupled-grid", default="32x512")
    parser.add_argument("--front-nr", type=int, default=8)
    parser.add_argument("--front-nz-ladder", default="256;512;1024")
    parser.add_argument("--groups-list", default="2,4,8")
    parser.add_argument("--front-t-hot-ev", type=float, default=60.0)
    parser.add_argument("--front-t-cold-ev", type=float, default=3.0)
    parser.add_argument("--front-rho-gcc", type=float, default=1.0)
    parser.add_argument("--skip-coupled", action="store_true")
    parser.add_argument("--skip-front", action="store_true")
    parser.add_argument("--production-audit-tier", choices=("none", "A", "B"), default="A")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "skip a leg's subprocess when <out-root>/<label>/run.log exists "
            "AND run_info reports termination_reason=t_end_reached; the "
            "existing outputs are then re-read and gated normally "
            "(same-binary discipline is the caller's responsibility)"
        ),
    )
    parser.add_argument("--verbosity", default="quiet")
    args = parser.parse_args()

    groups_list = [int(v) for v in args.groups_list.split(",") if v.strip()]
    summary: dict[str, Any] = {
        "ladder_row": LADDER_ROW,
        "dimension": "2D_RZ",
        "groups_list": groups_list,
        "runs": {},
        "ode_absolute_metrics": {
            "gated": False,
            "caveat": (
                "two_state-mode absolute comparison vs the FLD-CED ODE "
                "reference is REPORTED only (adjudication Q6); binding "
                "collapse comparator is the same-binary R0 grey run"
            ),
        },
    }
    all_pass = True

    if not args.skip_coupled:
        cnr, cnz = (int(v) for v in args.coupled_grid.lower().split("x"))
        r0 = run_deck(args, "R0_grey_coupled", coupled_env(1), cnr, cnz)
        summary["runs"]["R0"] = sanitize(r0)
        r0_ok = hygiene_pass(r0)
        summary["r0_hygiene_pass"] = r0_ok
        all_pass &= r0_ok
        if r0.get("snapshot") is not None:
            summary["r0_wave12"] = wave12_report(r0["snapshot"])
        mg_runs: dict[int, dict[str, Any]] = {}
        coupled_pass = r0_ok
        for g in groups_list:
            run = run_deck(args, f"R2_g{g}_flat_coupled", coupled_env(g), cnr, cnz)
            summary["runs"][f"R2_g{g}"] = sanitize(run)
            mg_runs[g] = run
            ok = hygiene_pass(run) and assert_effective_groups(run, g)
            summary[f"r2_g{g}_hygiene_pass"] = ok
            coupled_pass &= ok
            if ok and r0.get("snapshot") is not None and run.get("snapshot") is not None:
                comp = compare_profiles(run["snapshot"], r0["snapshot"])
                summary[f"r2_g{g}_collapse"] = comp
                summary[f"r2_g{g}_wave12"] = wave12_report(run["snapshot"])
                gate = all(
                    v["eps_L2"] <= COUPLED_EPS_L2_GATE
                    and v["eps_Linf"] <= COUPLED_EPS_LINF_GATE
                    for v in comp.values()
                )
                summary[f"r2_g{g}_collapse_pass"] = gate
                coupled_pass &= gate
            else:
                summary[f"r2_g{g}_collapse_pass"] = False
                coupled_pass = False
        ladder_pass = True
        gl = [g for g in groups_list if mg_runs.get(g, {}).get("snapshot") is not None]
        for a_idx in range(len(gl)):
            for b_idx in range(a_idx + 1, len(gl)):
                ga, gb = gl[a_idx], gl[b_idx]
                comp = compare_profiles(mg_runs[ga]["snapshot"], mg_runs[gb]["snapshot"])
                worst = max(v["eps_L2"] for v in comp.values())
                summary[f"ladder_eps_L2_g{ga}_g{gb}"] = worst
                ladder_pass &= worst <= LADDER_PAIRWISE_L2_GATE
        summary["coupled_ladder_pass"] = ladder_pass and len(gl) == len(groups_list)
        summary["coupled_collapse_pass"] = coupled_pass
        all_pass &= coupled_pass and summary["coupled_ladder_pass"]

    if not args.skip_front:
        means = freq_dep_band_means(args.front_t_hot_ev, args.front_rho_gcc)
        summary["front_grey_comparators"] = means
        front_nzs = [int(v) for v in args.front_nz_ladder.split(";") if v.strip()]
        mid_nz = front_nzs[len(front_nzs) // 2]
        greys: dict[str, dict[str, Any]] = {}
        for tag, kappa in (
            ("kP", means["kappa_P_cm2g"]),
            ("kR", means["kappa_R_cm2g"]),
        ):
            run = run_deck(
                args,
                f"R3g_{tag}_front",
                front_env(1, "flat", kappa_override=kappa),
                args.front_nr,
                mid_nz,
            )
            summary["runs"][f"R3g_{tag}"] = sanitize(run)
            greys[tag] = run
            all_pass &= hygiene_pass(run)
        front_z: dict[int, float] = {}
        front_pass = True
        for g in groups_list:
            run = run_deck(
                args, f"R3_g{g}_structured_front", front_env(g, "structured"),
                args.front_nr, mid_nz,
            )
            summary["runs"][f"R3_g{g}"] = sanitize(run)
            ok = hygiene_pass(run) and assert_effective_groups(run, g)
            front_pass &= ok
            if ok and run.get("snapshot") is not None:
                fm = front_metrics(
                    run["snapshot"], args.front_t_hot_ev, args.front_t_cold_ev
                )
                front_z[g] = fm["front_z_cm"]
                dep: dict[str, Any] = dict(fm)
                for tag, grun in greys.items():
                    if grun.get("snapshot") is not None:
                        gm = front_metrics(
                            grun["snapshot"], args.front_t_hot_ev, args.front_t_cold_ev
                        )
                        dep[f"dz_front_vs_{tag}_cm"] = (
                            fm["front_z_cm"] - gm["front_z_cm"]
                        )
                        dep[f"eps_Te_vs_{tag}"] = compare_profiles(
                            run["snapshot"], grun["snapshot"]
                        )["Te"]
                summary[f"r3_g{g}_departure"] = sanitize(dep)
        cauchy_pass = True
        if all(g in front_z for g in (2, 4, 8)):
            d24 = abs(front_z[2] - front_z[4])
            d48 = abs(front_z[4] - front_z[8])
            summary["front_cauchy_d24_cm"] = d24
            summary["front_cauchy_d48_cm"] = d48
            dz_cell = 0.24 / mid_nz
            floor = 0.1 * dz_cell
            if d24 <= floor and d48 <= floor:
                summary["front_cauchy_note"] = "both deltas below resolution floor"
            else:
                cauchy_pass = d48 <= FRONT_CAUCHY_CONTRACTION_GATE * max(d24, TINY)
        else:
            cauchy_pass = False
        summary["front_cauchy_pass"] = cauchy_pass and front_pass
        rich: dict[int, float] = {}
        for nz in front_nzs:
            run = run_deck(
                args, f"R4_g4_structured_front_nz{nz}",
                front_env(4, "structured"), args.front_nr, nz,
            )
            summary["runs"][f"R4_nz{nz}"] = sanitize(run)
            if hygiene_pass(run) and run.get("snapshot") is not None:
                rich[nz] = front_metrics(
                    run["snapshot"], args.front_t_hot_ev, args.front_t_cold_ev
                )["front_z_cm"]
        if len(rich) == len(front_nzs) and len(front_nzs) >= 3:
            zs = [rich[nz] for nz in sorted(rich)]
            d1 = abs(zs[0] - zs[1])
            d2 = abs(zs[1] - zs[2])
            summary["richardson_front_z"] = {str(nz): rich[nz] for nz in sorted(rich)}
            summary["richardson_delta_coarse_cm"] = d1
            summary["richardson_delta_fine_cm"] = d2
            summary["richardson_contracting"] = bool(d2 <= d1 + TINY)
            if d2 > TINY and d1 > TINY:
                summary["richardson_observed_order"] = math.log2(max(d1, TINY) / max(d2, TINY))
        else:
            summary["richardson_contracting"] = False
        summary["front_departure_recorded"] = front_pass and len(front_z) == len(groups_list)
        all_pass &= summary["front_departure_recorded"] and summary["front_cauchy_pass"]
        all_pass &= bool(summary.get("richardson_contracting", False))

    if args.skip_coupled and args.skip_front:
        # Anti-vacuity: no phase ran, nothing was verified.
        summary["skipped_everything"] = True
        all_pass = False
    summary["all_checks_passed"] = bool(all_pass)
    out_path = Path(args.summary_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sanitize(summary), indent=2), encoding="utf-8")
    print(f"[run_i2_2d_rz_mg_collapse] all_checks_passed={all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
