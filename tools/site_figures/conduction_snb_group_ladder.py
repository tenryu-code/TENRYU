"""SNB global groups, primitive, and local source weights.

24 edges intervals: beta_hat[k]=0.1*200**(k/24), k=0..24;
maximum temperature 2 keV, local temperature ratios 1 and 0.25.
w(b)=b**4*exp(-b)/24; P(b)=1-exp(-b)*(b**4+4*b**3+
12*b**2+24*b+24)/24; xi[g]=P(b[g])-P(b[g-1]).
Display 920x640, output DPI 200; top domain [0.08,25], bottom
domain [0.05,100]. Highlight group 12. Derivative probes 0.5,1,2,5,10
use centered differences with h=1e-4, relative tolerance 1e-6;
P(0) tolerance 1e-15 and telescoping tolerance 1e-14.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

W, H, DPI = 920, 640, 200
N_GROUPS, BETA_MIN, BETA_MAX, MAX_TE_KEV = 24, 0.1, 20.0, 2.0
TEMPERATURE_RATIOS = (1.0, 0.25)
HIGHLIGHT_GROUP = 12
PROBES = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
FD_STEP, DERIVATIVE_TOL, ZERO_TOL, SUM_TOL = 1e-4, 1e-6, 1e-15, 1e-14
BLUE, ORANGE = '#2563eb', '#ea580c'
OUTPUT = Path('docs/site/assets/conduction-snb-group-ladder.png')
warnings.filterwarnings('error', message='Glyph .* missing')
plt.rcParams.update({'font.family': ['Hiragino Sans', 'DejaVu Sans'],
    'font.size': 10, 'mathtext.fontset': 'stix', 'axes.linewidth': 0.9,
    'axes.edgecolor': '#475569', 'axes.labelcolor': '#1e293b',
    'text.color': '#1e293b', 'xtick.color': '#475569', 'ytick.color': '#475569',
    'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def primitive(b):
    b = np.asarray(b)
    return 1 - np.exp(-b) * (b**4 + 4*b**3 + 12*b**2 + 24*b + 24) / 24


def spectrum(b):
    return b**4 * np.exp(-b) / 24


def check(label, value, passed):
    print(f'{label}: {value} {"PASS" if passed else "FAIL"}')
    if not passed:
        raise SystemExit(1)


def main():
    edges = BETA_MIN * (BETA_MAX/BETA_MIN)**(np.arange(N_GROUPS+1)/N_GROUPS)
    check('P(0), tolerance 1e-15', primitive(0), abs(primitive(0)) <= ZERO_TOL)
    for b in PROBES:
        fd = (primitive(b+FD_STEP)-primitive(b-FD_STEP))/(2*FD_STEP)
        err = abs(fd-spectrum(b))/spectrum(b)
        check(f'dP/db at {b:g}, relative error < 1e-6', f'{err:.12g}', err < DERIVATIVE_TOL)
    weights = []
    for ratio in TEMPERATURE_RATIOS:
        p = primitive(edges/ratio)
        xi = np.diff(p)
        weights.append(xi)
        check(f'Te/maxTe={ratio:g}, sum xi and endpoint difference (tol 1e-14)',
              f'{xi.sum():.16g}, {p[-1]-p[0]:.16g}', abs(xi.sum()-(p[-1]-p[0])) <= SUM_TOL)

    fig = plt.figure(figsize=(W/100, H/100), dpi=DPI)
    top = fig.add_axes([0.075, 0.754, 0.885, 0.115])
    top.set_xscale('log')
    top.set_xlim(0.08, 25)
    top.set_ylim(0, 1)
    top.set_yticks([])
    for k in range(N_GROUPS):
        top.axvspan(edges[k], edges[k+1], color=('#dbeafe', '#eff6ff')[k%2],
                    ec='#cbd5e1', lw=0.6)
    for g in (1, 12, 24):
        top.text(np.sqrt(edges[g-1]*edges[g]), 0.5, f'$g={g}$',
                 ha='center', va='center', fontsize=8.5)
    top.set_xlabel(r'$\hat\beta = E/(k_B\max T_e)$', labelpad=4)
    sec = top.secondary_xaxis('top', functions=(lambda x: MAX_TE_KEV*x, lambda x: x/MAX_TE_KEV))
    sec.set_xlabel(r'$E$ [keV] for $\max T_e = 2$ keV', labelpad=5)
    fig.text(0.075, 0.968, r'global group edges in $\hat\beta = E/(k_B \max T_e)$',
             fontsize=11, fontweight='semibold', va='top')
    fig.text(0.96, 0.681, r'edges rebuilt from $\max T_e$ every conduction step',
             fontsize=8.5, ha='right', color='#64748b')

    ax = fig.add_axes([0.075, 0.145, 0.435, 0.425])
    ax.set_xscale('log')
    ax.set_xlim(0.05, 100)
    ax.set_ylim(0, 1.035)
    ax.set_xlabel(r'$\beta = E/(k_B T_e)$')
    ax.set_ylabel(r'$P(\beta)$')
    ax.set_title('source weights from $P$ at the\nlocal $\\beta = E/(k_B T_e)$',
                 fontsize=11, fontweight='semibold', pad=14)
    beta = np.geomspace(0.05, 100, 1000)
    twin = ax.twinx()
    twin.plot(beta, spectrum(beta), color='#94a3b8', lw=1.2)
    twin.set_ylim(0, 0.24)
    twin.set_ylabel(r'$w(\beta)=\beta^4e^{-\beta}/24$', labelpad=7)
    twin.tick_params(labelsize=8.5)
    ax.plot(beta, primitive(beta), color='#1e40af', lw=1.8)
    for ratio, color, bx in zip(TEMPERATURE_RATIOS, (BLUE, ORANGE), (3.0, 14.0)):
        local = edges/ratio
        p = primitive(local)
        ax.plot(local, p, '|', color=color, ms=7, mew=1.2)
        lo, hi = p[HIGHLIGHT_GROUP-1:HIGHLIGHT_GROUP+1]
        ax.plot([bx, bx], [lo, hi], color=color, lw=1.4)
        for val in (lo, hi):
            ax.plot([bx/1.13, bx*1.13], [val, val], color=color, lw=1.4)
        ax.text(bx*1.24, max((lo+hi)/2, 0.055), r'$\xi_{12}$', color=color, va='center')
    ax.grid(axis='y', color='#cbd5e1', lw=0.5, alpha=0.6)

    bars = fig.add_axes([0.705, 0.145, 0.267, 0.425])
    groups = np.arange(1, N_GROUPS+1)
    bars.bar(groups-0.20, weights[0], width=0.4, color=BLUE)
    bars.bar(groups+0.20, weights[1], width=0.4, color=ORANGE)
    bars.set_xlim(0, 25)
    bars.set_xticks([1, 6, 12, 18, 24])
    bars.set_xlabel('$g$')
    bars.set_ylabel(r'$\xi_g = P(\beta_g) - P(\beta_{g-1})$', labelpad=6)
    bars.set_title(r'$\xi_g$ per group', fontsize=11, fontweight='semibold', pad=14)
    fig.text(0.70, 0.031, 'exact primitive differences, no quadrature', fontsize=8.5)
    fig.text(0.09, 0.031, r'$T_e = \max T_e$', color=BLUE)
    fig.text(0.29, 0.031, r'$T_e = 0.25\max T_e$', color=ORANGE)
    fig.savefig(OUTPUT, dpi=DPI, facecolor='white')
    with Image.open(OUTPUT) as im:
        check('PNG size', im.size, im.size == (2*W, 2*H))
        print(f'WROTE {OUTPUT} {im.width}x{im.height}')
    plt.close(fig)


if __name__ == '__main__':
    main()
