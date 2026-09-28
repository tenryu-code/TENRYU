"""One computed corona fan rotated onto two CBET ports and cell crossings.

Reference rays have b=+-0.16,+-0.48,+-0.80,+-1.12; rc=c=1,
L=0.35, EPS_O=0.02, R_OUT=1+L*ln(50), RK4 dt=0.001.
The corona_rays model integrates dv/dt=-grad(nhat)/2 with
nhat=(exp(-(r-1)/L)-EPS_O)/(1-EPS_O), zero outside R_OUT.
P=exp(-K integral nhat**2 dt); K gives axial inbound depth 1.
Ports i,j rotate this fan by -35,+35 degrees, respectively; paths
are shown only at r<R_OUT+0.5. Ports sit at incoming radius 3.05;
rotation arrows use radius 2.75. The highlighted shell is
1.35<r<1.60. Exact polyline segment intersections use midpoint
KD trees only to cull impossible pairs. Invariant tolerances are
energy 1e-6, Bouguer 1e-5, turning 0.002, rotation radii 1e-12;
at least four intersections must lie inside the shell.
Canvas 900x560 display pixels at dpi=200 (1800x1120 output).
Left/right panel widths are 304/540 display pixels, with equal
aspect. Port bars have half-length 0.43, and crossings have area 22.
"""

import sys
sys.dont_write_bytecode = True
import warnings
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, FancyArrowPatch
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D
from PIL import Image
from corona_rays import R_OUT, trace_ray, rotate, print_checks, absorption_constant

W, H, DPI = 900, 560, 200
IMPACTS = (-1.12, -0.80, -0.48, -0.16, 0.16, 0.48, 0.80, 1.12)
ANGLES = np.deg2rad([-35.0, 35.0])
SHELL_INNER, SHELL_OUTER = 1.35, 1.60
DRAW_RADIUS = R_OUT+0.5
PORT_RADIUS, ARC_RADIUS, PORT_HALF_LENGTH = 3.05, 2.75, 0.43
BLUE, ORANGE, PURPLE = '#2563eb', '#ea580c', '#7c3aed'
OUTPUT = Path('docs/site/assets/cbet-port-section-geometry.png')


def cross2(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def intersections(paths_i, paths_j):
    a0 = np.vstack([p[:-1] for p in paths_i])
    a1 = np.vstack([p[1:] for p in paths_i])
    b0 = np.vstack([p[:-1] for p in paths_j])
    b1 = np.vstack([p[1:] for p in paths_j])
    am, bm = (a0+a1)/2, (b0+b1)/2
    ar = np.linalg.norm(a1-a0, axis=1)/2
    br = np.linalg.norm(b1-b0, axis=1)/2
    candidates = cKDTree(bm).query_ball_point(am, ar+np.max(br)+1e-14)
    ai = np.repeat(np.arange(len(am)), [len(v) for v in candidates])
    bi = np.array([j for values in candidates for j in values], dtype=int)
    u, v = a1[ai]-a0[ai], b1[bi]-b0[bi]
    offset = b0[bi]-a0[ai]
    determinant = cross2(u, v)
    valid = np.abs(determinant) > 1e-20
    ta = np.full(len(ai), np.inf)
    tb = np.full(len(ai), np.inf)
    ta[valid] = cross2(offset[valid], v[valid])/determinant[valid]
    tb[valid] = cross2(offset[valid], u[valid])/determinant[valid]
    valid &= (ta >= 0) & (ta <= 1) & (tb >= 0) & (tb <= 1)
    return a0[ai[valid]]+ta[valid, None]*u[valid]


def main():
    warnings.filterwarnings('error', message=r'Glyph .* missing')
    plt.rcParams.update({'font.family': ['Hiragino Sans', 'DejaVu Sans'],
                         'font.size': 10, 'mathtext.fontset': 'stix',
                         'axes.linewidth': 0.9, 'axes.edgecolor': '#475569',
                         'axes.labelcolor': '#1e293b', 'text.color': '#1e293b',
                         'xtick.color': '#475569', 'ytick.color': '#475569',
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    reference = [trace_ray(b) for b in IMPACTS]
    print(f'K={absorption_constant():.15g}')
    checks = [print_checks(c) for _, _, c in reference]
    all_paths = [np.column_stack((a['z'], a['R'])) for a, _, _ in reference]
    rotated = [[rotate(p, angle) for p in all_paths] for angle in ANGLES]
    rotation_error = max(np.max(np.abs(np.linalg.norm(q, axis=1)-np.linalg.norm(p, axis=1)))
                         for fan in rotated for p, q in zip(all_paths, fan))
    ok = rotation_error < 1e-12
    checks.append(ok)
    print(f'rotation_radius_error={rotation_error:.15g} < 1e-12: {"PASS" if ok else "FAIL"}')
    crossings = intersections(*rotated)
    radii = np.linalg.norm(crossings, axis=1)
    shell_crossings = crossings[(radii > SHELL_INNER) & (radii < SHELL_OUTER)]
    ok = len(shell_crossings) >= 4
    checks.append(ok)
    print(f'crossings_inside_shell={len(shell_crossings)} >= 4: {"PASS" if ok else "FAIL"}')
    if not all(checks):
        raise SystemExit(1)
    fig = plt.figure(figsize=(W/100, H/100), dpi=DPI)
    left = fig.add_axes([18/W, 105/H, 304/W, 376/H], aspect='equal')
    right = fig.add_axes([342/W, 100/H, 540/W, 386/H], aspect='equal')
    left.set(xlim=(-3.60, 2.70), ylim=(-3.50, 4.29))
    right.set(xlim=(-3.85, 4.80), ylim=(-3.25, 2.933))
    for ax in (left, right):
        ax.set_axis_off()
        ax.add_patch(Circle((0, 0), R_OUT, fill=False, edgecolor='#94a3b8', lw=0.8))
        ax.add_patch(Circle((0, 0), 1, fill=False, edgecolor='#dc2626', lw=1.6,
                            linestyle=(0, (5, 3)), zorder=2))
    fig.text(170/W, 523/H, 'Reference trace (computed once)', ha='center',
             fontsize=11, fontweight='semibold', fontfamily='Hiragino Sans')
    fig.text(612/W, 523/H, 'Rigid rotation onto the ports', ha='center',
             fontsize=11, fontweight='semibold', fontfamily='Hiragino Sans')
    right.add_patch(Wedge((0, 0), SHELL_OUTER, 0, 360, width=SHELL_OUTER-SHELL_INNER,
                          facecolor='#ede9fe', edgecolor='none', alpha=0.8, zorder=0))

    def draw_fan(ax, paths, color):
        for p, (_, turn, _) in zip(paths, reference):
            for section, ls, lw in [(slice(None, turn+1), '-', 1.3),
                                    (slice(turn, None), '--', 1.1)]:
                leg = p[section]
                leg = leg[np.linalg.norm(leg, axis=1) < DRAW_RADIUS]
                ax.plot(leg[:, 0], leg[:, 1], color=color, lw=lw, ls=ls, zorder=3)
            ax.scatter(*p[turn], s=10, color=color, zorder=4)

    draw_fan(left, all_paths, '#334155')
    left.plot([-2.93, -3.07, -3.07, -2.93], [-1.12, -1.12, 1.12, 1.12],
               color='#475569', lw=1)
    left.text(-3.32, 0, r'impact bins $\beta$', rotation=90, ha='center', va='center', fontsize=10)
    left.annotate('reference axis', xy=(-2.54, 0), xytext=(-3.32, 2.88), fontsize=10,
                  arrowprops=dict(arrowstyle='-', color='#475569', lw=0.8))
    left.add_patch(FancyArrowPatch((-2.97, 0), (-2.53, 0), arrowstyle='-|>',
                                   mutation_scale=10, color='#475569', lw=1.3, zorder=5))
    left.legend([Line2D([0], [0], color='#334155', lw=1.3),
                 Line2D([0], [0], color='#334155', lw=1.1, ls='--')],
                [r'inbound leg (sheet $\sigma=0$)', r'outbound leg (sheet $\sigma=1$)'],
                frameon=False, fontsize=8.5, loc='lower center', bbox_to_anchor=(0.5, -0.04))
    for angle, fan, color, symbol in zip(ANGLES, rotated, (BLUE, ORANGE), ('i', 'j')):
        draw_fan(right, fan, color)
        center = rotate(np.array([-PORT_RADIUS, 0.0]), angle)
        tangent = rotate(np.array([0.0, PORT_HALF_LENGTH]), angle)
        right.plot([center[0]-tangent[0], center[0]+tangent[0]],
                   [center[1]-tangent[1], center[1]+tangent[1]], color=color, lw=5, solid_capstyle='butt')
        right.text(center[0]-0.25, center[1]+(0.50 if angle < 0 else -0.62),
                    f'port ${symbol}$', color=color, ha='center')
        theta = np.linspace(np.pi, np.pi+angle, 80)
        path = MplPath(ARC_RADIUS*np.column_stack((np.cos(theta), np.sin(theta))))
        right.add_patch(FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=12,
                                        lw=1.3, color=PURPLE, zorder=5))
        mid = np.pi+angle*0.58
        right.text(3.13*np.cos(mid), 3.13*np.sin(mid), f'$R_{symbol}$', color=PURPLE,
                    ha='center', va='center', fontsize=10)
    right.plot([-2.86, -2.64], [0, 0], color='#94a3b8', lw=1.5, zorder=6)
    right.text(-4.0, -0.26, 'reference', fontsize=10, color='#64748b')
    right.scatter(shell_crossings[:, 0], shell_crossings[:, 1], s=22, color=PURPLE, zorder=7)
    right.annotate('cell $c$ (spherical shell)', xy=(1.38, -0.53), xytext=(1.80, -2.93),
                   color=PURPLE, fontsize=10,
                   arrowprops=dict(arrowstyle='-', lw=0.8, color=PURPLE))
    chosen = shell_crossings[np.argmax(shell_crossings[:, 0])]
    right.annotate("pair crossings: states\n$A=(i,\\sigma,\\beta)$ and\n"
                   "$B=(j,\\sigma',\\beta')$\nthat coexist in cell $c$\nare coupled once",
                   xy=chosen, xytext=(2.02, 1.52), fontsize=8.5, va='top',
                   arrowprops=dict(arrowstyle='-', color=PURPLE, lw=0.8))
    fig.text(350/W, 33/H,
             'One reference trace serves every port by rigid rotation;\nno port is traced again.',
             fontsize=8.5, color='#64748b', va='bottom')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(fig.texts)+list(left.texts)+list(right.texts)+list(left.get_legend().get_texts())
    bounds = [text.get_window_extent(renderer) for text in texts]
    margin = min(min(b.x0, b.y0, 2*W-b.x1, 2*H-b.y1) for b in bounds)/2
    print(f'text_canvas_margin={margin:.6f} display_px >= 12: {"PASS" if margin >= 12 else "FAIL"}')
    if margin < 12:
        raise SystemExit(1)
    fig.savefig(OUTPUT, dpi=DPI, facecolor='white')
    plt.close(fig)
    with Image.open(OUTPUT) as im:
        print(f'WROTE {OUTPUT} {im.width}x{im.height}')
        if im.size != (2*W, 2*H):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
