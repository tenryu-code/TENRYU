#!/usr/bin/env python3
"""S_N reference solver for linearized Su-Olson with convergence reporting.

Dimensionless model (xi = sigma*x, tau = sigma*c*t):
  dI_n/dtau + mu_n dI_n/dxi = (theta - I_n) + Q(xi,tau)
  eps*dtheta/dtau = V - theta
  V = (1/2) * sum_n w_n * I_n

BC:
  xi=0 reflecting: I(mu>0) = I(-mu)
  xi=L vacuum    : incoming I(mu<0 at xi=L) = 0

Numerics:
  - S_16 Gauss-Legendre quadrature
  - 1st-order upwind streaming
  - Strang splitting: stream(dt/2) -> collide(dt) -> stream(dt/2)
  - Collision sink treated implicitly
  - dt is auto-scaled from target CFL
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RunConfig:
    sn_order: int = 16
    epsilon: float = 1.0
    L: float = 20.0
    tau_end: float = 10.0
    source_xi0: float = 0.5
    source_tau0: float = 10.0
    source_s0: float = 0.2
    source_direction_factor: float = 1.0
    target_cfl: float = 0.4


@dataclass(frozen=True)
class SolveResult:
    n_x: int
    dxi: float
    dtau_base: float
    n_steps: int
    v_mean: np.ndarray
    theta: np.ndarray


def _gauss_legendre_sn(order: int) -> tuple[np.ndarray, np.ndarray]:
    mu, w = np.polynomial.legendre.leggauss(order)
    return mu.astype(np.float64), w.astype(np.float64)


def _sample_at_x(values: np.ndarray, dxi: float, probe_x: Iterable[float]) -> np.ndarray:
    n_x = values.size
    centers = (np.arange(n_x, dtype=np.float64) + 0.5) * dxi
    out: list[float] = []
    for x in probe_x:
        if x <= 0.0:
            out.append(float(values[0]))
        elif x >= centers[-1]:
            out.append(float(values[-1]))
        else:
            out.append(float(np.interp(x, centers, values)))
    return np.asarray(out, dtype=np.float64)


def _stream_upwind(
    I_in: np.ndarray,
    I_out: np.ndarray,
    mu: np.ndarray,
    pair: np.ndarray,
    dtau: float,
    dxi: float,
) -> None:
    m = mu.size
    for n in range(m):
        mu_n = mu[n]
        cfl = abs(mu_n) * dtau / dxi
        if mu_n > 0.0:
            left_in = I_in[pair[n], 0]
            I_out[n, 0] = I_in[n, 0] - cfl * (I_in[n, 0] - left_in)
            I_out[n, 1:] = I_in[n, 1:] - cfl * (I_in[n, 1:] - I_in[n, :-1])
        else:
            I_out[n, :-1] = I_in[n, :-1] + cfl * (I_in[n, 1:] - I_in[n, :-1])
            I_out[n, -1] = I_in[n, -1] + cfl * (0.0 - I_in[n, -1])


def _run_sn(cfg: RunConfig, n_x: int, case: str, mu: np.ndarray, w: np.ndarray) -> SolveResult:
    if case not in {"delta", "volume"}:
        raise ValueError(f"unknown case: {case}")

    m = mu.size
    dxi = cfg.L / float(n_x)
    dtau_base = cfg.target_cfl * dxi / float(np.max(np.abs(mu)))
    eps = cfg.epsilon

    pair = np.arange(m - 1, -1, -1, dtype=np.int64)
    I = np.zeros((m, n_x), dtype=np.float64)
    theta = np.zeros(n_x, dtype=np.float64)

    if case == "delta":
        I[:, 0] = 1.0 / dxi

    I_half = np.empty_like(I)
    I_coll = np.empty_like(I)
    I_next = np.empty_like(I)

    x_centers = (np.arange(n_x, dtype=np.float64) + 0.5) * dxi
    source_mask = x_centers <= (cfg.source_xi0 + 1.0e-14)
    q_dir = cfg.source_direction_factor * cfg.source_s0
    q_vec = np.where(source_mask, q_dir, 0.0).astype(np.float64)
    q_zero = np.zeros(n_x, dtype=np.float64)

    tau = 0.0
    step = 0
    while tau < cfg.tau_end - 1.0e-15:
        dt = min(dtau_base, cfg.tau_end - tau)
        dt_half = 0.5 * dt
        tau_mid = tau + dt_half

        if case == "volume" and tau_mid <= (cfg.source_tau0 + 1.0e-14):
            q = q_vec
        else:
            q = q_zero

        # Strang step 1: stream by dt/2.
        _stream_upwind(I, I_half, mu, pair, dt_half, dxi)

        # Strang step 2: collision/source by dt (implicit coupled absorption-emission).
        V_stream = 0.5 * np.sum(w[:, None] * I_half, axis=0)
        # For direction-independent isotropic source q_n = q, q_avg = q.
        q_avg = q
        denom = eps + dt * (eps + 1.0)
        theta = (theta * eps * (1.0 + dt) + dt * (V_stream + dt * q_avg)) / denom
        I_coll[:, :] = (I_half + dt * (theta[None, :] + q[None, :])) / (1.0 + dt)

        # Strang step 3: stream by dt/2.
        _stream_upwind(I_coll, I_next, mu, pair, dt_half, dxi)
        I, I_next = I_next, I

        tau += dt
        step += 1

    V_final = 0.5 * np.sum(w[:, None] * I, axis=0)
    return SolveResult(
        n_x=n_x,
        dxi=dxi,
        dtau_base=dtau_base,
        n_steps=step,
        v_mean=V_final,
        theta=theta,
    )


def _p1_theta_volume_center_talbot(
    tau: float,
    s00: float,
    xi0: float,
    dps: int = 50,
) -> float:
    try:
        import mpmath as mp
    except ModuleNotFoundError as exc:
        raise RuntimeError("P1 Talbot inversion requires mpmath.") from exc

    if tau <= 0.0:
        raise ValueError("tau must be > 0 for inverse Laplace")

    mp.mp.dps = max(dps, 30)
    tau_mp = mp.mpf(str(tau))
    s00_mp = mp.mpf(str(s00))
    xi0_mp = mp.mpf(str(xi0))

    def _theta_hat(s: mp.mpf | mp.mpc) -> mp.mpc:
        kappa = mp.sqrt(3.0 * s * (s + 2.0) / (s + 1.0))
        return (s00_mp / (s * s * (s + 2.0))) * (1.0 - mp.exp(-kappa * xi0_mp))

    inv = mp.invertlaplace(_theta_hat, tau_mp, method="talbot")
    return float(mp.re(inv))


def _print_probe_table(title: str, probe: list[float], theta: np.ndarray, vmean: np.ndarray) -> None:
    print(title)
    for x, th, vv in zip(probe, theta.tolist(), vmean.tolist()):
        print(f"  xi={x:>3.1f}: theta={th:.10f}, V_mean={vv:.10f}")
    print()


def main() -> None:
    cfg = RunConfig()
    probe = [0.0, 1.0, 3.0, 5.0]
    exact_delta_theta = np.array([0.7696, 0.6633, 0.3937, 0.2001], dtype=np.float64)
    n_x_levels = [2000, 4000, 8000]

    mu, w = _gauss_legendre_sn(cfg.sn_order)
    mu_max = float(np.max(np.abs(mu)))

    print("Su-Olson S_N reference (Strang split + CFL auto-scaling)")
    print(
        f"config: S_{cfg.sn_order}, eps={cfg.epsilon}, L={cfg.L}, "
        f"tau_end={cfg.tau_end}, target_CFL={cfg.target_cfl}, max|mu|={mu_max:.10f}"
    )
    print(
        "volume source: S0=0.2 on xi in [0,0.5], tau in [0,10], "
        "transport directional term = S0"
    )
    print()

    delta_results: list[SolveResult] = []
    volume_results: list[SolveResult] = []

    print("Convergence study (delta source):")
    for n_x in n_x_levels:
        res = _run_sn(cfg, n_x=n_x, case="delta", mu=mu, w=w)
        delta_results.append(res)
        th0 = float(_sample_at_x(res.theta, res.dxi, [0.0])[0])
        rel0 = abs(th0 - exact_delta_theta[0]) / max(abs(exact_delta_theta[0]), 1.0e-30)
        print(
            f"  n_x={n_x:4d}, dxi={res.dxi:.8f}, dtau={res.dtau_base:.8f}, "
            f"steps={res.n_steps:5d}, CFL_max={cfg.target_cfl:.3f}, "
            f"theta(0,10)={th0:.10f}, rel_err_vs_0.7696={rel0:.4%}"
        )
    print()

    print("Convergence study (volume source):")
    for n_x in n_x_levels:
        res = _run_sn(cfg, n_x=n_x, case="volume", mu=mu, w=w)
        volume_results.append(res)
        th0 = float(_sample_at_x(res.theta, res.dxi, [0.0])[0])
        print(
            f"  n_x={n_x:4d}, dxi={res.dxi:.8f}, dtau={res.dtau_base:.8f}, "
            f"steps={res.n_steps:5d}, CFL_max={cfg.target_cfl:.3f}, theta(0,10)={th0:.10f}"
        )
    print()

    delta_finest = delta_results[-1]
    volume_finest = volume_results[-1]

    delta_th_probe = _sample_at_x(delta_finest.theta, delta_finest.dxi, probe)
    delta_v_probe = _sample_at_x(delta_finest.v_mean, delta_finest.dxi, probe)
    volume_th_probe = _sample_at_x(volume_finest.theta, volume_finest.dxi, probe)
    volume_v_probe = _sample_at_x(volume_finest.v_mean, volume_finest.dxi, probe)

    delta_rel_err = np.abs(delta_th_probe - exact_delta_theta) / np.maximum(
        np.abs(exact_delta_theta), 1.0e-30
    )
    delta_ok = delta_rel_err[0] <= 0.05

    _print_probe_table(
        f"Delta-source probes at tau=10 (n_x={delta_finest.n_x}):",
        probe,
        delta_th_probe,
        delta_v_probe,
    )
    print("Delta-source comparison vs exact transport benchmark:")
    for x, th, ex, re in zip(probe, delta_th_probe, exact_delta_theta, delta_rel_err):
        print(f"  xi={x:>3.1f}: theta_SN={th:.10f}, theta_exact={ex:.10f}, rel_err={re:.4%}")
    print()

    _print_probe_table(
        f"Volume-source probes at tau=10 (n_x={volume_finest.n_x}):",
        probe,
        volume_th_probe,
        volume_v_probe,
    )

    ratio_0 = (
        volume_th_probe[0] / delta_th_probe[0]
        if abs(delta_th_probe[0]) > 1.0e-30
        else np.nan
    )
    print(f"ratio theta_volume/theta_delta at xi=0: {ratio_0:.10f}")
    print()

    if delta_ok:
        print("Delta check PASSED (|theta_SN(0,10)-0.7696|/0.7696 <= 5%).")
        print("S_N transport references are considered trustworthy for volume source.")
    else:
        print("Delta check FAILED (>5% from 0.7696).")
        print("Computing P1 Laplace-inversion estimate for volume-source theta(0,10).")
        theta_p1_talbot = _p1_theta_volume_center_talbot(
            tau=cfg.tau_end,
            s00=cfg.source_s0,
            xi0=cfg.source_xi0,
            dps=60,
        )
        print(f"  P1 Talbot estimate: theta(0,10) = {theta_p1_talbot:.10f}")
    print()

    print("Final reference values for cmd_verify.cpp (volume-source theta, tau=10):")
    print(
        "const std::vector<double> u_ref = {"
        + ", ".join(f"{v:.10f}" for v in volume_th_probe.tolist())
        + "};"
    )


if __name__ == "__main__":
    main()
