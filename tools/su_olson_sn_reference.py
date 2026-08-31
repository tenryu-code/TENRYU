#!/usr/bin/env python3
"""Independent S_N reference for the su_olson volume-source configuration.

Integrates the EXACT as-built model of examples/verification/su_olson.py run
in sn_transport mode (grey S_8 transport + constant-cv matter — NOT the
Su & Olson alpha*T^3 benchmark):

    du_m/dt + c mu_m du_m/dx + c sigma u_m = (c sigma a Te^4 + S(x)) / 2
    cv dTe/dt = c sigma (E - a Te^4),   E = sum_m w_m u_m

with sigma = 1/cm, cv = 548.8 erg/cc/eV, a = 137.2017 erg/cc/eV^4,
S = 3.2928e12 erg/cc/s for x <= 0.5 cm, reflecting inner face (x=0),
vacuum outer face (x=20), full-range Gauss-Legendre S_8 (sum w = 2),
t_end = 3.33e-10 s (tau = sigma*c*t ~ 10). Explicit donor-cell upwind
streaming on the deck's own 800-cell grid at CFL 0.4 with explicit
absorption-emission exchange; the printed Te/T_ref triple at
xi = {0.01, 1.0, 3.16228} is the reference table for the SN volume-source
validation (the FLD analogue lives in tools/su_olson_fld_reference.py).

This is an implementation-verification reference (independent discretization
of the same PDE system), not a model benchmark: the published Su & Olson
(1997) transport tables assume alpha*T^3 matter, which the runtime's
standard matter update does not implement.
"""

import numpy as np

c = 2.99792458e10
a_eV = 137.2017  # erg/cc/eV^4 (radiation constant in eV units)
sigma = 1.0
cv = 548.8
S0 = 3.2928e12
x0 = 0.5
L = 20.0
n = 800
t_end = 3.33e-10
Te0 = 1.0e-3
T_ref = 1.0
n_mu = 8


def main():
    dx = L / n
    xc = (np.arange(n) + 0.5) * dx
    mu, w = np.polynomial.legendre.leggauss(n_mu)  # full range, sum w = 2
    u = np.zeros((n_mu, n))  # E-normalized intensity: E = sum_m w_m u_m
    Te = np.full(n, Te0)
    dt = 0.4 * dx / c
    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps
    src = np.where(xc <= x0, S0, 0.0)
    for _ in range(n_steps):
        emis = c * sigma * a_eV * Te**4
        rhs = 0.5 * (emis + src)
        E = np.einsum("m,mn->n", w, u)
        new_u = np.empty_like(u)
        for m in range(n_mu):
            um = u[m]
            if mu[m] > 0.0:
                # upwind neighbor is i-1; x=0 reflects the -mu partner
                # (Gauss-Legendre nodes are symmetric: partner = n_mu-1-m).
                left = np.empty(n)
                left[0] = u[n_mu - 1 - m][0]
                left[1:] = um[:-1]
                stream = -c * mu[m] * (um - left) / dx
            else:
                # upwind neighbor is i+1; x=L vacuum: incoming = 0.
                right = np.empty(n)
                right[-1] = 0.0
                right[:-1] = um[1:]
                stream = -c * mu[m] * (right - um) / dx
            new_u[m] = um + dt * (stream - c * sigma * um + rhs)
        u = np.maximum(new_u, 0.0)
        E = np.einsum("m,mn->n", w, u)
        Te = Te + dt * c * sigma * (E - a_eV * Te**4) / cv
        np.maximum(Te, Te0, out=Te)
    E = np.einsum("m,mn->n", w, u)
    for xi in (0.01, 1.0, 3.16228):
        i = int(np.argmin(np.abs(xc - xi)))
        print(f"xi={xi:8.5f}  Te/T_ref={Te[i] / T_ref:.8f}  E={E[i]:.6e}")


if __name__ == "__main__":
    main()
