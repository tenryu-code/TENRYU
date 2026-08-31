#!/usr/bin/env python3
"""Independent reference for the su_olson FLD verification gate.

Integrates the EXACT model the verify deck configures (grey unlimited
diffusion + constant-cv matter — NOT the Su & Olson alpha*T^3 benchmark):

    dE/dt  = d/dx( c/(3 sigma) dE/dx ) + S(x,t) + c*sigma*(a*Te^4 - E)
    cv dTe/dt = c*sigma*(E - a*Te^4)

with sigma = 1/cm, cv = 548.8 erg/cc/eV, a = 137.2017 erg/cc/eV^4,
S = 3.2928e12 erg/cc/s for x <= 0.5 cm, reflecting inner face, vacuum outer
face (half-range leak flux c*E/2), domain 20 cm, t_end = 3.33e-10 s
(tau = sigma*c*t ~ 10). Explicit FTCS on the deck's own 800-cell grid with
a diffusion-stable dt; the printed Te/T_ref triple at xi = {0.01, 1.0,
3.16228} is the u_ref table in cmd_verify.cpp run_su_olson_verify.

This is an implementation-verification reference (independent discretization
of the same PDE system), not a model benchmark: the published Su & Olson
(1997) tables assume alpha*T^3 matter, which the runtime's standard FLD
matter update does not implement (2026-08-30 status; see the verify comment).
"""

import numpy as np

c = 2.99792458e10
a_eV = 137.2017  # erg/cc/eV^4 (radiation constant in eV units, matches core::constants::a_eV)
sigma = 1.0
cv = 548.8
S0 = 3.2928e12
x0 = 0.5
L = 20.0
n = 800
t_end = 3.33e-10
Te0 = 1.0e-3
T_ref = 1.0


def main():
    dx = L / n
    xc = (np.arange(n) + 0.5) * dx
    E = np.zeros(n)
    Te = np.full(n, Te0)
    D = c / (3.0 * sigma)
    dt = 0.2 * dx * dx / (2.0 * D)
    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps
    src = np.where(xc <= x0, S0, 0.0)
    for _ in range(n_steps):
        flux = np.zeros(n + 1)  # face fluxes, positive = +x
        flux[1:-1] = -D * (E[1:] - E[:-1]) / dx
        flux[0] = 0.0                # reflect
        flux[-1] = 0.5 * c * E[-1]   # vacuum half-range leak
        dEdt = -(flux[1:] - flux[:-1]) / dx + src
        exch = c * sigma * (a_eV * Te**4 - E)  # +into radiation, -into matter
        E = E + dt * (dEdt + exch)
        Te = Te - dt * exch / cv
        np.maximum(Te, Te0, out=Te)
        np.maximum(E, 0.0, out=E)
    for xi in (0.01, 1.0, 3.16228):
        i = int(np.argmin(np.abs(xc - xi)))
        print(f"xi={xi:8.5f}  Te/T_ref={Te[i] / T_ref:.8f}  E={E[i]:.6e}")


if __name__ == "__main__":
    main()
