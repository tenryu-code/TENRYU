"""Laser chords, Bouguer turning and IB deposition in a spherical corona.

The corona_rays model uses rc=c=1, L=0.35, EPS_O=0.02,
R_OUT=1+0.35*ln(50), dt=0.001 and dv/dt=-grad(nhat)/2;
nhat=(exp(-(r-1)/L)-EPS_O)/(1-EPS_O), zero outside R_OUT.
K is calibrated to unit inbound axial optical depth; P=exp(-integral
K*nhat**2 dt). Rays have b=0,0.45,0.80,1.15,1.55. Line width is
0.7+2.3*P pt; deposition is sampled every 0.14 arc-length units with
area 5+110*rate/max(rate) pt**2. Axial return offset is 0.035.
Canvas 900x560 display pixels, dpi=200 (1800x1120 output).
View z=[-4.70,6.10], R=[-3.16,3.56], equal aspect.
Numerical tolerances are energy 1e-6, Bouguer 1e-5, turning 0.002;
the axial minimum is in [0.999,1.001], all other roots exceed 1.
"""

import sys
sys.dont_write_bytecode = True
import warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.lines import Line2D
from PIL import Image
from corona_rays import R_OUT, density, trace_ray, print_checks, absorption_constant

W, H, DPI = 900, 560, 200
IMPACTS = (0.0, 0.45, 0.80, 1.15, 1.55)
MARKER_SPACING = 0.14
RETURN_OFFSET = 0.035
X_MIN, X_MAX, Y_MAX = -4.70, 6.10, 3.56
Y_MIN = Y_MAX-(X_MAX-X_MIN)*H/W
OUTPUT = Path('docs/site/assets/laser-chord-geometry.png')
BLUE, ORANGE, PURPLE, RED = '#2563eb', '#ea580c', '#7c3aed', '#dc2626'


def main():
    warnings.filterwarnings('error', message=r'Glyph .* missing')
    plt.rcParams.update({'font.family': ['Hiragino Sans', 'DejaVu Sans'],
                         'font.size': 10, 'mathtext.fontset': 'stix',
                         'axes.linewidth': 0.9, 'axes.edgecolor': '#475569',
                         'axes.labelcolor': '#1e293b', 'text.color': '#1e293b',
                         'xtick.color': '#475569', 'ytick.color': '#475569',
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    rays = [trace_ray(b) for b in IMPACTS]
    print(f'K={absorption_constant():.15g}')
    results = [print_checks(c) for _, _, c in rays]
    axial_depth = rays[0][2]['inbound_optical_depth']
    ok = abs(axial_depth-1.0) < 1e-12
    print(f'axial_inbound_optical_depth={axial_depth:.15g}; abs(error)<1e-12: {"PASS" if ok else "FAIL"}')
    if not all(results) or not ok:
        raise SystemExit(1)
    fig = plt.figure(figsize=(W/100, H/100), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(X_MIN, X_MAX), ylim=(Y_MIN, Y_MAX), aspect='equal')
    ax.set_axis_off()
    xx, yy = np.meshgrid(np.linspace(-R_OUT, R_OUT, 800), np.linspace(-R_OUT, R_OUT, 800))
    rr = np.hypot(xx, yy)
    shade = np.ma.masked_where((rr < 1) | (rr > R_OUT), density(rr))
    cmap = LinearSegmentedColormap.from_list('corona', ['white', '#bfdbfe'])
    ax.imshow(shade, extent=(-R_OUT, R_OUT, -R_OUT, R_OUT), origin='lower',
              vmin=0, vmax=1, cmap=cmap, alpha=0.85, zorder=0)
    ax.add_patch(Circle((0, 0), 1, facecolor='#e5e7eb', edgecolor='none', zorder=1))
    ax.add_patch(Circle((0, 0), R_OUT, fill=False, edgecolor='#94a3b8', lw=0.8))
    ax.add_patch(Circle((0, 0), 1, fill=False, edgecolor=RED, lw=1.6, linestyle=(0, (5, 3)), zorder=3))
    ax.axhline(0, color='#94a3b8', lw=0.8, ls='--', zorder=2)
    ax.text(-4.5, 0.12, 'beam axis', fontsize=10, color='#64748b')
    ax.text(0, 0.45, r'overdense  ($n_e > n_c$)', ha='center', zorder=10,
            bbox=dict(facecolor='#e5e7eb', edgecolor='none', pad=0.5))
    ax.text(2.40, 1.95, 'underdense corona', color='#1d4ed8')
    ax.annotate(r'critical surface  $n_e = n_c$', xy=(0.78, -0.625), xytext=(1.1, -0.92),
                color=RED, fontsize=10, arrowprops=dict(arrowstyle='-', color=RED, lw=0.8))
    maximum = max(np.max(a['dep_rate']) for a, _, _ in rays)

    def draw_leg(points, power, rate, inside, dashed=False, thin=False):
        segments = np.stack((points[:-1], points[1:]), axis=1)
        widths = 0.7+2.3*(power[:-1]+power[1:])/2
        if thin:
            widths *= 0.75
        ax.add_collection(LineCollection(segments, colors=BLUE, linewidths=widths,
                                         zorder=4))
        if dashed:
            # Mask pieces by cumulative arc length so the dash pattern survives
            # the short, individually power-weighted LineCollection segments.
            arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            collection = ax.collections[-1]
            visible = np.mod((arc[:-1]+arc[1:])/2, 0.075) < 0.045
            collection.set_segments(segments[visible])
            collection.set_linewidths(widths[visible])
        indices = np.flatnonzero(inside)
        if len(indices) > 1:
            p = points[indices]
            arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
            samples = np.arange(MARKER_SPACING/2, arc[-1], MARKER_SPACING)
            sx = np.interp(samples, arc, p[:, 0])
            sy = np.interp(samples, arc, p[:, 1])
            sr = np.interp(samples, arc, rate[indices])
            ax.scatter(sx, sy, s=5+110*sr/maximum, c=ORANGE, edgecolors='none', zorder=5)

    for b, (a, turn, c) in zip(IMPACTS, rays):
        p = np.column_stack((a['z'], a['R']))
        inside = a['nhat'] > 0
        if b == 0:
            draw_leg(p[:turn+1], a['P'][:turn+1], a['dep_rate'][:turn+1], inside[:turn+1])
            end = np.flatnonzero((np.arange(len(p)) >= turn) & (p[:, 0] >= -3.3))[-1]
            ret = p[turn:end+1].copy()
            ret[:, 1] += RETURN_OFFSET
            draw_leg(ret, a['P'][turn:end+1], a['dep_rate'][turn:end+1], inside[turn:end+1], True, True)
            ax.add_patch(FancyArrowPatch((-3.10, RETURN_OFFSET), (-3.3, RETURN_OFFSET),
                                         arrowstyle='-|>', mutation_scale=10, lw=0.9, color=BLUE, zorder=7))
            ax.scatter([-1], [0], s=30, facecolor='white', edgecolor=RED, zorder=8)
        else:
            draw_leg(p, a['P'], a['dep_rate'], inside)
            ax.scatter(*p[turn], s=20, color=PURPLE, zorder=8)
            exit_idx = c['corona_slice'].stop
            # Choose a visible segment beyond the corona on the outgoing leg.
            visible = np.flatnonzero((np.arange(len(p)) >= exit_idx) &
                                    (p[:, 0] > X_MIN+0.15) & (p[:, 0] < X_MAX-0.15) &
                                    (p[:, 1] < Y_MAX-0.15))
            idx = visible[min(140, len(visible)-1)]
            unit = np.array([a['vz'][idx], a['vR'][idx]])
            unit /= np.linalg.norm(unit)
            ax.add_patch(FancyArrowPatch(p[idx]-0.1*unit, p[idx]+0.08*unit,
                                         arrowstyle='-|>', mutation_scale=10, color=BLUE, lw=1, zorder=6))
        ax.add_patch(FancyArrowPatch((-3.0, b), (-2.82, b), arrowstyle='-|>',
                                     mutation_scale=10, color=BLUE, lw=1.2, zorder=6))
    ax.text(-4.45, -0.55,
            '$b = 0$: reaches $n_c$ and is reflected\n(default characteristic integration);\n'
            'the leapfrog march terminates it', fontsize=8.5, color='#334155', va='top')
    a, turn, _ = rays[2]
    point = np.array([a['z'][turn], a['R'][turn]])
    radial = point/np.linalg.norm(point)
    tangent = np.array([a['vz'][turn], a['vR'][turn]])
    tangent /= np.linalg.norm(tangent)
    ax.plot([0, point[0]], [0, point[1]], color=PURPLE, lw=0.9, ls='--', zorder=6)
    ax.scatter(*point, s=58, facecolor='white', edgecolor=PURPLE, lw=1.3, zorder=9)
    corner = np.array([point-0.12*radial, point-0.12*radial+0.12*tangent, point+0.12*tangent])
    ax.plot(corner[:, 0], corner[:, 1], color=PURPLE, lw=0.8, zorder=7)
    ax.text(*(0.68*point+0.24*tangent), '$r_B$', color=PURPLE, fontsize=10,
            ha='center', va='center', zorder=11)
    ax.annotate(r'Bouguer turning point: $n(r_B)\,r_B = b$', xy=point, xytext=(0.15, 1.10),
                color=PURPLE, fontsize=10,
                arrowprops=dict(arrowstyle='-', color=PURPLE, lw=0.8))
    ax.add_patch(FancyArrowPatch((-3.05, 0), (-3.05, 0.80), arrowstyle='<->',
                                 mutation_scale=10, lw=1, color='#475569', zorder=9))
    ax.text(-3.20, 0.39, '$b$', ha='right', va='center')
    handles = [Line2D([0], [0], color=BLUE, lw=2),
               Line2D([0], [0], marker='o', color=ORANGE, ls='none', markersize=5),
               Line2D([0], [0], color=RED, lw=1.6, ls='--'),
               Line2D([0], [0], marker='o', color=PURPLE, ls='none', markersize=4)]
    labels = ['ray (width ∝ carried power $P$)',
              'IB deposition along the path (area ∝ local rate)',
              'critical surface', 'turning point $r_B$']
    ax.legend(handles, labels, fontsize=8.5, frameon=False, loc='lower right',
              bbox_to_anchor=(0.975, 0.14), borderaxespad=0, labelspacing=0.35)
    ax.text(0.025, 0.027,
            'Model corona $n_e/n_c \\approx \\exp[-(r-r_c)/L]$, $L = 0.35\\,r_c$; '
            'paths from $d\\mathbf{v}/dt = -\\frac{c^2}{2}\\nabla(n_e/n_c)$\n'
            'keep $n\\,r\\sin\\alpha = b$ constant.',
            transform=ax.transAxes, fontsize=8.5, color='#64748b', va='bottom')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(ax.texts)+list(ax.get_legend().get_texts())
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
