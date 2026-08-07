#!/usr/bin/env python3
"""1D spherical Lagrangian (staggered VNR) reference for I1-B CASE=D.

V&V instrument for the I1-B Tier-2 gate (see
docs/validation/2d_rz/I1/i1b_tier2_gate_evidence_20260703.md). Env knobs:
T_GAS_EV (cushion temperature), P_PEAK_MBAR (override the deck-consistent
auto drive), FOOT_MBAR (two-step foot+main pulse), T_H_NS, T_END_NS,
NSCALE (resolution scale; self-convergence 0.40% per doubling at N_GAS=400),
ETA, RHO_GAS, RHO_SHELL. Offline tool: never called at runtime.

Exact deck parameters from 2d_rz_i1b_finite_cr_shell_hotspot.py with the
canonical pinned radii. Ideal gas gamma=5/3 single material, equal initial
total pressure, zero initial velocity, trapezoidal ablation-pressure drive
on the outer boundary. Outputs CR_V(t) over the initially-gas region,
kinetic/internal energy split, and conversion efficiency at peak.
"""
import math
import numpy as np

# --- deck parameters (canonical run) ---
EV = 1.602176634e-12
AMU = 1.66053906660e-24
A_AMU, ZBAR, GAMMA = 1.0, 1.0, 5.0 / 3.0
R_HS, R_G, R_S, R_OUTER, S_MAX = 0.005, 0.035, 0.0455, 0.049, 0.052
import os as _os
REF_GEOM = _os.environ.get("REF_GEOM", "spherical").strip().lower()
if REF_GEOM not in ("spherical", "planar", "cylindrical"):
    raise ValueError("REF_GEOM must be 'spherical', 'planar', or 'cylindrical'")
PLANAR = REF_GEOM == "planar"
CYLINDRICAL = REF_GEOM == "cylindrical"
if CYLINDRICAL:
    if "REF_CYL_R_INNER_CM" not in _os.environ:
        raise ValueError("REF_CYL_R_INNER_CM is required when REF_GEOM=cylindrical")
    R_CYL_INNER = float(_os.environ["REF_CYL_R_INNER_CM"])
else:
    R_CYL_INNER = 0.0
A0 = 1.0
RHO_GAS = float(_os.environ.get("RHO_GAS", "1.0e-4"))
RHO_SHELL = float(_os.environ.get("RHO_SHELL", "0.25"))
RHO_ABL, RHO_EXT = 1.05, RHO_GAS
T_GAS_EV = float(_os.environ.get("T_GAS_EV", "1.0e3"))
P_BASE = RHO_GAS * EV * (ZBAR + 1.0) * T_GAS_EV / (A_AMU * AMU)  # equal-P init
ETA = float(__import__("os").environ.get("ETA", "0.5"))
if PLANAR:
    V_G0 = A0 * R_G
elif CYLINDRICAL:
    V_G0 = math.pi * ((R_CYL_INNER + R_G)**2 - R_CYL_INNER**2)
else:
    V_G0 = 4.0 / 3.0 * math.pi * R_G**3
C_G = R_G / R_HS
DELTA_U_G = 1.5 * P_BASE * V_G0 * (C_G * C_G - 1.0)
if PLANAR:
    V_SHELL = A0 * (R_S - R_G)
    V_ABL = A0 * (R_OUTER - R_S)
elif CYLINDRICAL:
    V_SHELL = math.pi * ((R_CYL_INNER + R_S)**2 - (R_CYL_INNER + R_G)**2)
    V_ABL = math.pi * ((R_CYL_INNER + R_OUTER)**2 - (R_CYL_INNER + R_S)**2)
else:
    V_SHELL = 4.0 / 3.0 * math.pi * (R_S**3 - R_G**3)
    V_ABL = 4.0 / 3.0 * math.pi * (R_OUTER**3 - R_S**3)
M_ACTIVE = RHO_SHELL * V_SHELL + RHO_ABL * V_ABL
U0 = math.sqrt(2.0 * DELTA_U_G / (ETA * M_ACTIVE))
AREAL = RHO_SHELL * (R_S - R_G) + RHO_ABL * (R_OUTER - R_S)
import os
if PLANAR or CYLINDRICAL:
    T_R = 0.20e-9
    T_H = 1.20e-9
    T_D = 0.15e-9
else:
    T_R = 0.25e-9
    T_H = float(os.environ.get("T_H_NS", "1.0")) * 1e-9
    T_D = 0.25e-9
DURATION = 0.5 * T_R + (T_H - T_R) + 0.5 * T_D
if PLANAR or CYLINDRICAL:
    P_PEAK = 60.0e12
else:
    P_PEAK = AREAL * U0 / DURATION
if "P_PEAK_MBAR" in os.environ:
    P_PEAK = float(os.environ["P_PEAK_MBAR"]) * 1e12
T_END = float(os.environ.get("T_END_NS", "4.0")) * 1e-9

FOOT = float(_os.environ.get("FOOT_MBAR", "0")) * 1e12
def p_drive(t):
    if FOOT > 0.0:
        tn = t * 1e9
        if tn < 0.20:
            return FOOT * tn / 0.20
        if tn < 0.70:
            return FOOT
        if tn < 1.05:
            return FOOT + (P_PEAK - FOOT) * (tn - 0.70) / 0.35
        if tn < 1.596:
            return P_PEAK
        if tn < 1.846:
            return P_PEAK * (1.0 - (tn - 1.596) / 0.25)
        return 0.0
    if t < T_R:
        return P_PEAK * t / T_R
    if t < T_H:
        return P_PEAK
    if t < T_H + T_D:
        return P_PEAK * (1.0 - (t - T_H) / T_D)
    return 0.0

# --- mesh: per-region uniform radial zoning ---
import os as _os2
_SC = float(_os2.environ.get("NSCALE", "1.0"))
N_GAS, N_SH, N_AB, N_EX = int(400*_SC), int(240*_SC), int(120*_SC), max(int(40*_SC),8)
_UNIFORM_N = int(_os2.environ.get("REF_UNIFORM_N", "0"))
if _UNIFORM_N > 0:
    # 2D-equivalent zoning probe: uniform radial edges across the whole
    # domain (the 2D butterfly resolves ~20 radial units total), so the
    # surface cell mixes buffer+ablator exactly as the coarse 2D IC does.
    edges = np.linspace(0.0, S_MAX, _UNIFORM_N + 1)
else:
    edges = np.concatenate([
        np.linspace(0.0, R_G, N_GAS + 1),
        np.linspace(R_G, R_S, N_SH + 1)[1:],
        np.linspace(R_S, R_OUTER, N_AB + 1)[1:],
        np.linspace(R_OUTER, S_MAX, N_EX + 1)[1:],
    ])
r = edges.copy()
if CYLINDRICAL:
    r = R_CYL_INNER + r
N = len(r) - 1
IF_GAS = int(np.searchsorted(edges, R_G * (1.0 - 1e-12)))  # gas-interface edge index
if CYLINDRICAL:
    rc0 = 0.5 * (edges[:-1] + edges[1:])
else:
    rc0 = 0.5 * (r[:-1] + r[1:])
_SMOOTH_W = float(_os2.environ.get("REF_SMOOTH_WIDTH_CM", "0.0"))
def _c2(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))
def _trans(a, b, rr, r0, w):
    if w <= 0.0:
        return np.where(rr <= r0, a, b)
    return a + (b - a) * _c2((rr - (r0 - w)) / (2.0 * w))
if _SMOOTH_W > 0.0:
    # Deck-matched layered IC: c2-smoothstep transitions of width
    # 2*SMOOTH_WIDTH centered at each interface (the 2D deck's _transition).
    rho = _trans(RHO_GAS, RHO_SHELL, rc0, R_G, _SMOOTH_W)
    rho = np.where(rc0 > 0.5 * (R_G + R_S),
                   _trans(RHO_SHELL, RHO_ABL, rc0, R_S, _SMOOTH_W), rho)
    rho = np.where(rc0 > 0.5 * (R_S + R_OUTER),
                   _trans(RHO_ABL, RHO_EXT, rc0, R_OUTER, _SMOOTH_W), rho)
else:
    rho = np.where(rc0 <= R_G, RHO_GAS,
          np.where(rc0 <= R_S, RHO_SHELL,
          np.where(rc0 <= R_OUTER, RHO_ABL, RHO_EXT)))
if PLANAR:
    vol = A0 * (r[1:] - r[:-1])
elif CYLINDRICAL:
    vol = math.pi * (r[1:]**2 - r[:-1]**2)
else:
    vol = 4.0 / 3.0 * math.pi * (r[1:]**3 - r[:-1]**3)
m = rho * vol
e = P_BASE / ((GAMMA - 1.0) * rho)          # specific internal energy
u = np.zeros(N + 1)
gas = rc0 <= R_G
M_GAS = m[gas].sum()
V_GAS0 = vol[gas].sum()
RHOBAR0 = M_GAS / V_GAS0
U_GAS0 = (m[gas] * e[gas]).sum()

# node masses (half-cells each side)
mn = np.zeros(N + 1)
mn[:-1] += 0.5 * m
mn[1:] += 0.5 * m

_RV_FILE = _os2.environ.get("REF_RV_FILE", "")
_RV_T = None
_RV_R = None
if _RV_FILE:
    _tbl = np.loadtxt(_RV_FILE)
    _RV_T = np.concatenate([[0.0], _tbl[:, 0]])
    if PLANAR:
        _V0_2D = A0 * S_MAX
        _RV_R = np.concatenate([[_V0_2D], _tbl[:, 1]]) / A0
    elif CYLINDRICAL:
        _V0_2D = math.pi * ((R_CYL_INNER + S_MAX)**2 - R_CYL_INNER**2)
        _RV_R = np.sqrt(R_CYL_INNER**2 + np.concatenate([[_V0_2D], _tbl[:, 1]]) / math.pi)
    else:
        _V0_2D = 4.0 / 3.0 * math.pi * S_MAX**3
        _RV_R = (3.0 * np.concatenate([[_V0_2D], _tbl[:, 1]]) / (4.0 * math.pi)) ** (1.0 / 3.0)

C1, C2 = 0.5, 4.0   # match the 2D run's AV profile (linear, quadratic)
CFL = 0.25
t, dt = 0.0, 1.0e-15
W_drive = 0.0   # cumulative work done ON the capsule by the drive [erg]
W_iface = 0.0   # cumulative work done ON the gas across the gas/shell interface [erg]
peak = (0.0, 0.0, 0.0, 0.0)   # CR, t, U_gas_gain, K_tot
K_at_peak_time = 0.0
Kmax = 0.0
hist = []
step = 0
while t < T_END:
    if PLANAR:
        volc = A0 * (r[1:] - r[:-1])
    elif CYLINDRICAL:
        volc = math.pi * (r[1:]**2 - r[:-1]**2)
    else:
        volc = 4.0 / 3.0 * math.pi * (r[1:]**3 - r[:-1]**3)
    rhoc = m / volc
    P = (GAMMA - 1.0) * rhoc * e
    cs = np.sqrt(GAMMA * P / rhoc)
    du = u[1:] - u[:-1]
    q = np.where(du < 0.0, rhoc * (C1 * cs * np.abs(du) + C2 * du * du), 0.0)
    dr = r[1:] - r[:-1]
    dtn = CFL * np.min(dr / (cs + np.abs(du) + 1.0e-30))
    dt = min(dtn, 1.05 * dt, T_END - t + 1.0e-30)
    # momentum (staggered): interior nodes
    Pi = P + q
    if PLANAR:
        A = np.full_like(r, A0)
    elif CYLINDRICAL:
        A = 2.0 * math.pi * r
    else:
        A = 4.0 * math.pi * r**2
    f = np.zeros(N + 1)
    f[1:-1] = -(Pi[1:] - Pi[:-1]) * A[1:-1]
    f[-1] = -(p_drive(t) - Pi[-1]) * A[-1]
    unew = u + dt * f / np.maximum(mn, 1.0e-300)
    if _RV_T is not None and t < _RV_T[-1]:
        # Verdict-#7 E3 replay: prescribe the outer node to follow the 2D
        # run's volume-equivalent radius R_V(t); after the table ends the
        # boundary is released to the free (p_drive) dynamics above.
        rv_now = np.interp(t, _RV_T, _RV_R)
        rv_next = np.interp(t + dt, _RV_T, _RV_R)
        unew[-1] = (rv_next - rv_now) / dt
    unew[0] = 0.0
    rnew = r + dt * unew
    # positivity guard
    if np.any(np.diff(rnew) <= 0.0):
        dt *= 0.5
        continue
    if PLANAR:
        voln = A0 * (rnew[1:] - rnew[:-1])
    elif CYLINDRICAL:
        voln = math.pi * (rnew[1:]**2 - rnew[:-1]**2)
    else:
        voln = 4.0 / 3.0 * math.pi * (rnew[1:]**3 - rnew[:-1]**3)
    e = e - Pi * (voln - volc) / m
    e = np.maximum(e, 1.0e-30)
    # work histories (midpoint areas): drive on the outer face, shell on gas
    if PLANAR:
        W_drive += p_drive(t) * A0 * (-(unew[-1])) * dt
        W_iface += Pi[max(IF_GAS - 1, 0)] * A0 * (-(unew[IF_GAS])) * dt
    elif CYLINDRICAL:
        r_out_mid = 0.5 * (r[-1] + rnew[-1])
        W_drive += p_drive(t) * 2.0 * math.pi * r_out_mid * (-(unew[-1])) * dt
        ri_mid = 0.5 * (r[IF_GAS] + rnew[IF_GAS])
        W_iface += Pi[max(IF_GAS - 1, 0)] * 2.0 * math.pi * ri_mid * (-(unew[IF_GAS])) * dt
    else:
        r_out_mid = 0.5 * (r[-1] + rnew[-1])
        W_drive += p_drive(t) * 4.0 * math.pi * r_out_mid**2 * (-(unew[-1])) * dt
        ri_mid = 0.5 * (r[IF_GAS] + rnew[IF_GAS])
        W_iface += Pi[max(IF_GAS - 1, 0)] * 4.0 * math.pi * ri_mid**2 * (-(unew[IF_GAS])) * dt
    r, u, t = rnew, unew, t + dt
    step += 1
    if step % 200 == 0 or t >= T_END:
        Vg = voln[gas].sum()
        CR = (RHOBAR0 / (M_GAS / Vg)) ** (-1.0 / 3.0)
        Ugain = (m[gas] * e[gas]).sum() - U_GAS0
        K = 0.5 * (mn * u * u).sum()
        Kmax = max(Kmax, K)
        r_iface = r[IF_GAS] if IF_GAS < len(r) else r[-1]
        hist.append((t, CR, Ugain, K, r_iface, W_drive, W_iface))
        if CR > peak[0]:
            peak = (CR, t, Ugain, K)

CR_p, t_p, Ug_p, K_p = peak
if PLANAR:
    print(
        f"REF_GEOM=planar  A0={A0:.1f} cm2  "
        f"P_BASE={P_BASE:.4e} dyn/cm2  P_PEAK={P_PEAK:.4e} ({P_PEAK/1e12:.1f} Mbar)"
    )
    print(f"1D planar peak: CR_V={CR_p:.3f} at t={t_p*1e9:.3f} ns")
    print("trajectory (t_ns, CR_V, X_iface_um, W_drive, W_iface):")
    for row in hist[:: max(1, len(hist)//40)]:
        tt, cr, _, _, ri, wd, wi = row
        print(f"  {tt*1e9:6.3f}  {cr:7.3f}  Xi={ri*1e4:.2f}  Wd={wd:.4e}  Wi={wi:.4e}")
elif CYLINDRICAL:
    print(f"[ref1d] REF_GEOM=cylindrical r_inner_cm={R_CYL_INNER}")
    print(f"P_BASE={P_BASE:.4e} dyn/cm2  P_PEAK={P_PEAK:.4e} ({P_PEAK/1e12:.1f} Mbar)")
    print(f"1D cylindrical peak: CR_V={CR_p:.3f} at t={t_p*1e9:.3f} ns")
    print("trajectory (t_ns, CR_V, R_iface_um, W_drive, W_iface):")
    for row in hist[:: max(1, len(hist)//40)]:
        tt, cr, _, _, ri, wd, wi = row
        print(f"  {tt*1e9:6.3f}  {cr:7.3f}  Ri={ri*1e4:.2f}  Wd={wd:.4e}  Wi={wi:.4e}")
else:
    print(f"P_BASE={P_BASE:.4e} dyn/cm2  P_PEAK={P_PEAK:.4e} ({P_PEAK/1e12:.1f} Mbar)")
    print(f"U0={U0/1e5:.1f} km/s  M_active={M_ACTIVE:.4e} g  DELTA_U_G={DELTA_U_G:.4e} erg")
    print(f"1D peak: CR_V={CR_p:.3f} at t={t_p*1e9:.3f} ns")
    print(f"U_gas_gain at peak={Ug_p:.4e} erg  (CR7 needs {DELTA_U_G:.4e})")
    print(f"K_max={Kmax:.4e} erg  eta_eff=U_gain/K_max={Ug_p/max(Kmax,1e-300):.3f}")
    print("trajectory (t_ns, CR_V, R_iface_um, W_drive, W_iface):")
    for row in hist[:: max(1, len(hist)//40)]:
        tt, cr, ug, kk, ri, wd, wi = row
        print(f"  {tt*1e9:6.3f}  {cr:7.3f}  Ug={ug:.3e}  K={kk:.3e}  Ri={ri*1e4:.2f}  Wd={wd:.4e}  Wi={wi:.4e}")
