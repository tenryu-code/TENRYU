"""Deterministic corona characteristics and IB absorption in units rc = c = 1.

nhat = (exp(-(r-1)/L)-EPS_O)/(1-EPS_O), L=0.35, EPS_O=0.02,
R_OUT=1+L*log(1/EPS_O); nhat=0 outside.  dx/dt=v and
dv/dt=-grad(nhat)/2 are advanced with classical RK4, DT=0.001.
Entry is analytic from z=-(R_OUT+0.9), R=b, v=(1,0). Boundary
steps are bisected to 1e-12; vacuum exit ends at r=R_OUT+1.0.
P=exp(-K integral nhat**2 dt), deposition=K*nhat**2*P. K makes
the b=0 entry-to-minimum-radius optical depth exactly 1. Energy,
angular momentum and turning-radius tolerances are 1e-6, 1e-5,
and 0.002; the axial minimum must lie in [0.999,1.001].
"""

from functools import lru_cache
import numpy as np
from scipy.integrate import cumulative_trapezoid

L = 0.35
EPS_O = 0.02
R_OUT = 1.0 + L * np.log(1.0 / EPS_O)
DT = 1e-3
ENTRY_EXTENSION = 0.9
EXIT_EXTENSION = 1.0
BOUNDARY_TOL = 1e-12
ENERGY_TOL = 1e-6
BOUGUER_TOL = 1e-5
TURN_TOL = 2e-3


def density(r):
    r = np.asarray(r)
    return np.where(r <= R_OUT,
                    np.expm1((R_OUT - r) / L) * EPS_O / (1 - EPS_O), 0.0)


def _rhs(y):
    r = np.hypot(y[0], y[1])
    # The smooth interior extension is used by RK stages at the boundary;
    # accepted corona steps end on or inside the exact boundary.
    force = np.exp(-(r - 1) / L) / (2 * L * (1 - EPS_O))
    return np.array([y[2], y[3], force*y[0]/r, force*y[1]/r])


def _rk4(y, h):
    k1 = _rhs(y)
    k2 = _rhs(y + 0.5*h*k1)
    k3 = _rhs(y + 0.5*h*k2)
    k4 = _rhs(y + h*k3)
    return y + h*(k1 + 2*k2 + 2*k3 + k4)/6


@lru_cache(maxsize=None)
def _geometry(b):
    if abs(b) >= R_OUT:
        raise ValueError('The specified reference rays must enter the corona.')
    start_z = -(R_OUT + ENTRY_EXTENSION)
    entry_z = -np.sqrt(R_OUT**2 - b**2)
    entry_t = entry_z - start_z
    tv = np.arange(0.0, entry_t, DT)
    vacuum = np.column_stack((start_z+tv, np.full_like(tv, b),
                             np.ones_like(tv), np.zeros_like(tv)))
    y = np.array([entry_z, b, 1.0, 0.0])
    states, times = [y.copy()], [entry_t]
    for _ in range(30000):
        h = DT
        yn = _rk4(y, h)
        exiting = np.hypot(*yn[:2]) > R_OUT
        if exiting:
            lo, hi = 0.0, DT
            for _ in range(80):
                h = 0.5*(lo+hi)
                yn = _rk4(y, h)
                residual = np.hypot(*yn[:2]) - R_OUT
                if abs(residual) < BOUNDARY_TOL and residual <= 0:
                    break
                if residual > 0:
                    hi = h
                else:
                    lo = h
            else:
                raise RuntimeError('Boundary bisection failed.')
        times.append(times[-1]+h)
        states.append(yn.copy())
        y = yn
        if exiting:
            break
    else:
        raise RuntimeError('Ray did not leave the corona.')
    corona = np.array(states)
    ct = np.array(times)
    x, v = y[:2], y[2:]
    speed = np.linalg.norm(v)
    direction = v / speed
    distance = -np.dot(x, direction) + np.sqrt(
        np.dot(x, direction)**2 + (R_OUT+EXIT_EXTENSION)**2-np.dot(x,x))
    sv = np.append(np.arange(DT, distance, DT), distance)
    outgoing = np.column_stack((x + sv[:, None]*direction,
                                np.broadcast_to(v, (len(sv), 2))))
    all_states = np.vstack((vacuum, corona, outgoing))
    t = np.concatenate((tv, ct, ct[-1]+sv/speed))
    i0, i1 = len(tv), len(tv)+len(ct)
    return t, all_states, slice(i0, i1)


@lru_cache(maxsize=1)
def absorption_constant():
    t, y, cs = _geometry(0.0)
    tc, yc = t[cs], y[cs]
    nc = density(np.linalg.norm(yc[:, :2], axis=1))
    turning = np.argmin(np.linalg.norm(yc[:, :2], axis=1))
    integral = np.trapezoid(nc[:turning+1]**2, tc[:turning+1])
    return 1.0 / integral


def bouguer_root(b):
    if b == 0:
        return 1.0
    lo, hi = 1.0, R_OUT
    for _ in range(80):
        mid = (lo+hi)/2
        if mid*np.sqrt(max(0.0, 1-density(mid))) < abs(b):
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def trace_ray(b):
    """Return named arrays, global turning index, and numerical checks."""
    t, y, cs = _geometry(float(b))
    r = np.linalg.norm(y[:, :2], axis=1)
    nhat = density(r)
    # Vacuum segments have exactly zero absorption, including the endpoint.
    nhat[:cs.start] = 0
    nhat[cs.stop:] = 0
    optical_depth = absorption_constant()*cumulative_trapezoid(nhat**2, t, initial=0)
    power = np.exp(-optical_depth)
    turning = int(np.argmin(r))
    root = bouguer_root(b)
    energy = np.max(np.abs(np.sum(y[cs, 2:]**2, axis=1)+nhat[cs]-1))
    angular = np.abs(y[cs, 0]*y[cs, 3]-y[cs, 1]*y[cs, 2])
    bouguer = np.max(np.abs(angular-abs(b)))
    checks = dict(b=b, r_B=root, min_r=r[turning], energy_error=energy,
                  bouguer_error=bouguer, turning_error=abs(r[turning]-root),
                  absorbed_fraction=1-power[-1], corona_slice=cs,
                  inbound_optical_depth=optical_depth[turning])
    arrays = dict(t=t, z=y[:, 0], R=y[:, 1], vz=y[:, 2], vR=y[:, 3],
                  nhat=nhat, P=power,
                  dep_rate=absorption_constant()*nhat**2*power)
    return arrays, turning, checks


def print_checks(checks):
    b = checks['b']
    print(f"b={b:.2f} r_B={checks['r_B']:.12f} min_r={checks['min_r']:.12f} "
          f"absorbed_fraction={checks['absorbed_fraction']:.12f}")
    results = []
    for name, tolerance in [('bouguer_error', BOUGUER_TOL),
                            ('energy_error', ENERGY_TOL), ('turning_error', TURN_TOL)]:
        value = checks[name]
        ok = value < tolerance
        results.append(ok)
        print(f"  {name}={value:.12g} < {tolerance:g}: {'PASS' if ok else 'FAIL'}")
    if b == 0:
        ok = 0.999 <= checks['min_r'] <= 1.001
        print(f"  axial_min_r={checks['min_r']:.12f} in [0.999,1.001]: {'PASS' if ok else 'FAIL'}")
        results.append(ok)
    else:
        ok = checks['r_B'] > 1
        print(f"  r_B={checks['r_B']:.12f} > 1: {'PASS' if ok else 'FAIL'}")
        results.append(ok)
    return all(results)


def rotate(points, angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.asarray(points) @ np.array([[c, s], [-s, c]])
