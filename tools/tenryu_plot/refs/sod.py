"""Exact Sod Riemann reference solution."""

from __future__ import annotations

import math

import numpy as np


class SodExact:
    """Exact Riemann solution for a two-state ideal-gas tube at rest.

    Provenance: transcribed from struct SodExactSolution in
    src/drivers/cmd_verify.cpp and Toro, "Riemann Solvers and Numerical
    Methods for Fluid Dynamics", Ch. 4.
    """

    def __init__(
        self,
        rho_l: float = 1.0,
        p_l: float = 1.0,
        rho_r: float = 0.125,
        p_r: float = 0.1,
        gamma: float = 1.4,
    ) -> None:
        self.rho_l = rho_l
        self.p_l = p_l
        self.rho_r = rho_r
        self.p_r = p_r
        self.gamma = gamma
        self.p_star = 0.0
        self.u_star = 0.0
        self.solve()

    def side_function(
        self,
        p: float,
        rho_k: float,
        p_k: float,
        c_k: float,
    ) -> tuple[float, float]:
        g = self.gamma
        if p > p_k:
            a = 2.0 / ((g + 1.0) * rho_k)
            b = (g - 1.0) / (g + 1.0) * p_k
            q = math.sqrt(a / (p + b))
            f = (p - p_k) * q
            df = q * (1.0 - 0.5 * (p - p_k) / (p + b))
        else:
            f = (
                2.0
                * c_k
                / (g - 1.0)
                * ((p / p_k) ** ((g - 1.0) / (2.0 * g)) - 1.0)
            )
            df = (p / p_k) ** (-(g + 1.0) / (2.0 * g)) / (rho_k * c_k)
        return f, df

    def solve(self) -> None:
        g = self.gamma
        cl = math.sqrt(g * self.p_l / self.rho_l)
        cr = math.sqrt(g * self.p_r / self.rho_r)
        p = 0.5 * (self.p_l + self.p_r)
        for _ in range(100):
            fl, dfl = self.side_function(p, self.rho_l, self.p_l, cl)
            fr, dfr = self.side_function(p, self.rho_r, self.p_r, cr)
            dp = (fl + fr) / (dfl + dfr)
            p = max(p - dp, 1.0e-14 * self.p_l)
            if abs(dp) <= 1.0e-14 * p:
                break
        self.p_star = p
        fl, _ = self.side_function(p, self.rho_l, self.p_l, cl)
        fr, _ = self.side_function(p, self.rho_r, self.p_r, cr)
        self.u_star = 0.5 * (fr - fl)

    def rho_at(self, xi: float) -> float:
        g = self.gamma
        cl = math.sqrt(g * self.p_l / self.rho_l)
        cr = math.sqrt(g * self.p_r / self.rho_r)
        if xi <= self.u_star:
            if self.p_star > self.p_l:
                sl = -cl * math.sqrt(
                    (g + 1.0) / (2.0 * g) * self.p_star / self.p_l
                    + (g - 1.0) / (2.0 * g)
                )
                if xi < sl:
                    return self.rho_l
                return self.rho_l * (
                    (self.p_star / self.p_l + (g - 1.0) / (g + 1.0))
                    / ((g - 1.0) / (g + 1.0) * self.p_star / self.p_l + 1.0)
                )
            c_star = cl * (self.p_star / self.p_l) ** ((g - 1.0) / (2.0 * g))
            if xi < -cl:
                return self.rho_l
            if xi > self.u_star - c_star:
                return self.rho_l * (self.p_star / self.p_l) ** (1.0 / g)
            cf = 2.0 / (g + 1.0) * (cl - 0.5 * (g - 1.0) * xi)
            return self.rho_l * (cf / cl) ** (2.0 / (g - 1.0))
        if self.p_star > self.p_r:
            sr = cr * math.sqrt(
                (g + 1.0) / (2.0 * g) * self.p_star / self.p_r
                + (g - 1.0) / (2.0 * g)
            )
            if xi > sr:
                return self.rho_r
            return self.rho_r * (
                (self.p_star / self.p_r + (g - 1.0) / (g + 1.0))
                / ((g - 1.0) / (g + 1.0) * self.p_star / self.p_r + 1.0)
            )
        c_star_r = cr * (self.p_star / self.p_r) ** ((g - 1.0) / (2.0 * g))
        if xi > cr:
            return self.rho_r
        if xi < self.u_star + c_star_r:
            return self.rho_r * (self.p_star / self.p_r) ** (1.0 / g)
        cf = 2.0 / (g + 1.0) * (cr + 0.5 * (g - 1.0) * xi)
        return self.rho_r * (cf / cr) ** (2.0 / (g - 1.0))

    def u_at(self, xi: float) -> float:
        g = self.gamma
        cl = math.sqrt(g * self.p_l / self.rho_l)
        cr = math.sqrt(g * self.p_r / self.rho_r)
        if xi <= self.u_star:
            if self.p_star > self.p_l:
                sl = -cl * math.sqrt(
                    (g + 1.0) / (2.0 * g) * self.p_star / self.p_l
                    + (g - 1.0) / (2.0 * g)
                )
                if xi < sl:
                    return 0.0
                return self.u_star
            c_star = cl * (self.p_star / self.p_l) ** ((g - 1.0) / (2.0 * g))
            if xi < -cl:
                return 0.0
            if xi > self.u_star - c_star:
                return self.u_star
            return 2.0 / (g + 1.0) * (cl + xi)
        if self.p_star > self.p_r:
            sr = cr * math.sqrt(
                (g + 1.0) / (2.0 * g) * self.p_star / self.p_r
                + (g - 1.0) / (2.0 * g)
            )
            if xi > sr:
                return 0.0
            return self.u_star
        c_star_r = cr * (self.p_star / self.p_r) ** ((g - 1.0) / (2.0 * g))
        if xi > cr:
            return 0.0
        if xi < self.u_star + c_star_r:
            return self.u_star
        return 2.0 / (g + 1.0) * (xi - cr)

    def p_at(self, xi: float) -> float:
        g = self.gamma
        cl = math.sqrt(g * self.p_l / self.rho_l)
        cr = math.sqrt(g * self.p_r / self.rho_r)
        if xi <= self.u_star:
            if self.p_star > self.p_l:
                sl = -cl * math.sqrt(
                    (g + 1.0) / (2.0 * g) * self.p_star / self.p_l
                    + (g - 1.0) / (2.0 * g)
                )
                if xi < sl:
                    return self.p_l
                return self.p_star
            c_star = cl * (self.p_star / self.p_l) ** ((g - 1.0) / (2.0 * g))
            if xi < -cl:
                return self.p_l
            if xi > self.u_star - c_star:
                return self.p_star
            cf = 2.0 / (g + 1.0) * (cl - 0.5 * (g - 1.0) * xi)
            return self.p_l * (cf / cl) ** (2.0 * g / (g - 1.0))
        if self.p_star > self.p_r:
            sr = cr * math.sqrt(
                (g + 1.0) / (2.0 * g) * self.p_star / self.p_r
                + (g - 1.0) / (2.0 * g)
            )
            if xi > sr:
                return self.p_r
            return self.p_star
        c_star_r = cr * (self.p_star / self.p_r) ** ((g - 1.0) / (2.0 * g))
        if xi > cr:
            return self.p_r
        if xi < self.u_star + c_star_r:
            return self.p_star
        cf = 2.0 / (g + 1.0) * (cr + 0.5 * (g - 1.0) * xi)
        return self.p_r * (cf / cr) ** (2.0 * g / (g - 1.0))

    def profile(self, x, t: float, x0: float = 0.0) -> dict[str, np.ndarray]:
        x_arr = np.asarray(x, dtype=np.float64)
        rho = np.empty_like(x_arr, dtype=np.float64)
        u = np.empty_like(x_arr, dtype=np.float64)
        p = np.empty_like(x_arr, dtype=np.float64)
        if t <= 0.0:
            left = (x_arr - x0) <= 0.0
            rho[left] = self.rho_l
            rho[~left] = self.rho_r
            u[:] = 0.0
            p[left] = self.p_l
            p[~left] = self.p_r
            return {"rho": rho, "u": u, "p": p}

        xi = (x_arr - x0) / t
        for index, value in np.ndenumerate(xi):
            rho[index] = self.rho_at(float(value))
            u[index] = self.u_at(float(value))
            p[index] = self.p_at(float(value))
        return {"rho": rho, "u": u, "p": p}
