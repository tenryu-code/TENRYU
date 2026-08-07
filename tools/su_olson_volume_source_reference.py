#!/usr/bin/env python3
"""Compute Su-Olson references with a 1D P1 method-of-lines solver.

Model:
  dV/dtau = (1/3) d2V/dxi2 + (theta - V) + S0(xi,tau)
  eps * dtheta/dtau = V - theta

BC:
  xi=0   : dV/dxi = 0  (reflecting)
  xi=L   : V = 0       (vacuum)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SolverConfig:
    epsilon: float = 1.0
    xi0: float = 0.5
    tau0: float = 10.0
    tau_end: float = 10.0
    total_energy: float = 1.0
    dxi: float = 0.01
    dtau: float = 0.001
    L: float = 20.0

    @property
    def nx(self) -> int:
        # Include both boundaries: xi in [0, L]
        return int(round(self.L / self.dxi)) + 1

    @property
    def nt(self) -> int:
        return int(round(self.tau_end / self.dtau))

    @property
    def source_amplitude(self) -> float:
        return self.total_energy / (self.xi0 * self.tau0)


def _prepare_tridiagonal(cfg: SolverConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Unknowns are V[0..nx-2]. V[nx-1] at xi=L is fixed (Dirichlet) to 0.
    n = cfg.nx - 1
    dx2 = cfg.dxi * cfg.dxi
    eps = cfg.epsilon
    dt = cfg.dtau

    reaction = eps / (eps + dt)
    r = dt / 3.0
    off = -r / dx2
    diag_base = 1.0 + dt * reaction + 2.0 * r / dx2

    lower = np.full(n - 1, off, dtype=np.float64)
    diag = np.full(n, diag_base, dtype=np.float64)
    upper = np.full(n - 1, off, dtype=np.float64)

    # Neumann at xi=0: dV/dxi=0 => V[-1]=V[1], so d2V[0]=2*(V[1]-V[0])/dx^2.
    upper[0] = -2.0 * r / dx2

    return lower, diag, upper


def _factor_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = diag.size
    c_prime = np.empty(n - 1, dtype=np.float64)
    denom = np.empty(n, dtype=np.float64)

    denom[0] = diag[0]
    c_prime[0] = upper[0] / denom[0]
    for i in range(1, n - 1):
        denom[i] = diag[i] - lower[i - 1] * c_prime[i - 1]
        c_prime[i] = upper[i] / denom[i]
    denom[n - 1] = diag[n - 1] - lower[n - 2] * c_prime[n - 2]
    return c_prime, denom


def _solve_tridiagonal_factored(
    lower: np.ndarray,
    c_prime: np.ndarray,
    denom: np.ndarray,
    rhs: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    n = rhs.size
    y[0] = rhs[0] / denom[0]
    for i in range(1, n):
        y[i] = (rhs[i] - lower[i - 1] * y[i - 1]) / denom[i]

    x[n - 1] = y[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = y[i] - c_prime[i] * x[i + 1]
    return x


def _run_case(cfg: SolverConfig, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode not in {"volume_source", "delta_source"}:
        raise ValueError(f"Unknown mode: {mode}")

    n = cfg.nx - 1
    xi = np.linspace(0.0, cfg.L, cfg.nx, dtype=np.float64)

    V = np.zeros(n, dtype=np.float64)
    theta = np.zeros(n, dtype=np.float64)

    if mode == "delta_source":
        # Approximate half-space delta pulse at xi=0: V(0,0)=2*delta(xi)
        V[0] = 2.0 / cfg.dxi

    lower, diag, upper = _prepare_tridiagonal(cfg)
    c_prime, denom = _factor_tridiagonal(lower, diag, upper)

    rhs = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    V_new = np.empty(n, dtype=np.float64)
    theta_new = np.empty(n, dtype=np.float64)

    eps = cfg.epsilon
    dt = cfg.dtau
    reaction = eps / (eps + dt)

    source_profile = np.zeros(n, dtype=np.float64)
    source_mask = xi[:n] <= (cfg.xi0 + 1.0e-12)
    source_profile[source_mask] = cfg.source_amplitude

    for step in range(cfg.nt):
        tau_next = (step + 1) * dt
        if mode == "volume_source" and tau_next <= (cfg.tau0 + 1.0e-12):
            src = source_profile
        else:
            src = 0.0

        rhs[:] = V + dt * (reaction * theta + src)
        _solve_tridiagonal_factored(lower, c_prime, denom, rhs, y, V_new)
        theta_new[:] = (eps * theta + dt * V_new) / (eps + dt)

        V, V_new = V_new, V
        theta, theta_new = theta_new, theta

    # Reattach Dirichlet boundary node for reporting/interpolation.
    V_full = np.zeros(cfg.nx, dtype=np.float64)
    theta_full = np.zeros(cfg.nx, dtype=np.float64)
    V_full[:n] = V
    theta_full[:n] = theta
    return V_full, theta_full


def _sample(values: np.ndarray, cfg: SolverConfig, probe_xi: list[float]) -> list[float]:
    out: list[float] = []
    for x in probe_xi:
        idx = int(round(x / cfg.dxi))
        idx = min(max(idx, 0), values.size - 1)
        out.append(float(values[idx]))
    return out


def main() -> None:
    cfg = SolverConfig()
    probe = [0.0, 1.0, 3.0, 5.0]

    V_delta, th_delta = _run_case(cfg, mode="delta_source")
    V_vol, th_vol = _run_case(cfg, mode="volume_source")

    th_delta_probe = _sample(th_delta, cfg, probe)
    th_vol_probe = _sample(th_vol, cfg, probe)
    V_vol_probe = _sample(V_vol, cfg, probe)

    theta0_delta = th_delta_probe[0]
    theta0_volume = th_vol_probe[0]
    ratio = theta0_volume / theta0_delta if theta0_delta != 0.0 else math.nan
    rel_to_tab2 = abs(theta0_delta - 0.7696) / 0.7696

    print("Su-Olson P1 reference (finite-difference, fully implicit)")
    print(
        f"config: eps={cfg.epsilon}, xi0={cfg.xi0}, tau0={cfg.tau0}, "
        f"L={cfg.L}, dxi={cfg.dxi}, dtau={cfg.dtau}, nx={cfg.nx}, nt={cfg.nt}"
    )
    print()

    print("Delta-source proxy (initial V[0]=2/dxi):")
    print(f"  theta(0,10) = {theta0_delta:.10f}")
    print(f"  vs 0.7696 rel diff = {rel_to_tab2:.4%}")
    print()

    print("Volume-source case (S0=E_total/(xi0*tau0), E_total=1):")
    for x, th, vv in zip(probe, th_vol_probe, V_vol_probe):
        print(f"  xi={x:>3.1f}: theta={th:.10f}, V={vv:.10f}")
    print()
    print(f"ratio theta_volume/theta_delta at xi=0: {ratio:.10f}")


if __name__ == "__main__":
    main()
