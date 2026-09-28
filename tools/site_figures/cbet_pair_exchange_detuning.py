"""One CBET pair's signed energy ledger and odd ion-acoustic response.

Sender loses A, receiver gains A*omega_r/omega_s, and the IAW ledger
gets A*(1-omega_r/omega_s). P(g)=g*alpha/((g*alpha)**2+(1-g**2)**2),
alpha=0.2, g in [-2.6,2.6] on 10401 points. Operating detunings are
0.70,0.98,0.35,-0.45; y limits [-5.8,5.8]. The positive peak and
its actual half-maximum crossings are solved numerically. Checks:
oddness and P(1)=1/alpha errors <1e-12, FWHM within 20% of alpha.
Canvas is 940x400 display pixels, dpi=200 (1880x800 PNG). Diagram
boxes occupy x=[18,156], [198,354], [61,315]; plot bounds in display
pixels are [438,102,478,256]. Shift arrows start at y=-4.9 with
0.28 vertical spacing. All parameters are deterministic.
"""

import warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar, brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

W, H, DPI = 940, 400, 200
ALPHA = 0.2
G_MIN, G_MAX, GRID_SIZE = -2.6, 2.6, 10401
G0, STRONGER, WEAKER, REVERSED = 0.70, 0.98, 0.35, -0.45
Y_MIN, Y_MAX = -5.8, 5.8
OUTPUT = Path('docs/site/assets/cbet-pair-exchange-detuning.png')


def response(g):
    return g*ALPHA / ((g*ALPHA)**2+(1-g**2)**2)


def main():
    warnings.filterwarnings('error', message=r'Glyph .* missing')
    plt.rcParams.update({'font.family': ['Hiragino Sans', 'DejaVu Sans'],
                         'font.size': 10, 'mathtext.fontset': 'stix',
                         'axes.linewidth': 0.9, 'axes.edgecolor': '#475569',
                         'axes.labelcolor': '#1e293b', 'text.color': '#1e293b',
                         'xtick.color': '#475569', 'ytick.color': '#475569',
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    g = np.linspace(G_MIN, G_MAX, GRID_SIZE)
    odd_error = np.max(np.abs(response(g)+response(-g)))
    peak = minimize_scalar(lambda x: -response(x), bounds=(0.8, 1.2), method='bounded',
                           options={'xatol': 1e-14}).x
    half = response(peak)/2
    low = brentq(lambda x: response(x)-half, 0, peak, xtol=1e-14)
    high = brentq(lambda x: response(x)-half, peak, G_MAX, xtol=1e-14)
    fwhm = high-low
    checks = [odd_error < 1e-12, abs(response(1)-1/ALPHA) < 1e-12,
              abs(fwhm-ALPHA) <= 0.2*ALPHA]
    print(f'oddness_error={odd_error:.15g} < 1e-12: {"PASS" if checks[0] else "FAIL"}')
    print(f'P(1)={response(1):.15g}; error={abs(response(1)-1/ALPHA):.15g} < 1e-12: {"PASS" if checks[1] else "FAIL"}')
    print(f'FWHM={fwhm:.15g}; half_max_points=({low:.15g},{high:.15g}); '
          f'relative_error={abs(fwhm-ALPHA)/ALPHA:.15g} <= 0.2: {"PASS" if checks[2] else "FAIL"}')
    if not all(checks):
        raise SystemExit(1)
    fig = plt.figure(figsize=(W/100, H/100), dpi=DPI)
    diagram = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(0, H))
    diagram.set_axis_off()
    diagram.text(188, 372, 'One unordered pair, one exchange', fontsize=11,
                 fontweight='semibold', fontfamily='Hiragino Sans', ha='center')

    def box(x, y, width, height, fill):
        diagram.add_patch(FancyBboxPatch((x, y), width, height,
                          boxstyle='round,pad=0,rounding_size=8',
                          facecolor=fill, edgecolor='#475569', lw=1))

    box(18, 246, 138, 68, '#dbeafe')
    box(198, 246, 156, 68, '#dcfce7')
    diagram.text(87, 292, r'sender ($\omega_s$)', ha='center', va='center')
    diagram.text(87, 266, '$-A$', ha='center', va='center')
    diagram.text(276, 292, r'receiver ($\omega_r$)', ha='center', va='center')
    diagram.text(276, 266, r'$+A\,\omega_r/\omega_s$', ha='center', va='center')
    diagram.add_patch(FancyArrowPatch((156, 280), (198, 280), arrowstyle='-|>',
                      mutation_scale=12, lw=2.5, color='#2563eb', shrinkA=0, shrinkB=0))
    diagram.text(186, 333, 'wave action conserved (photon number)', fontsize=10, ha='center')
    box(61, 117, 254, 87, '#ede9fe')
    diagram.add_patch(FancyArrowPatch((177, 280), (177, 204), arrowstyle='-|>',
                      mutation_scale=12, lw=1.3, color='#475569', shrinkA=0, shrinkB=0))
    diagram.text(188, 184, 'ion-acoustic wave', ha='center', va='center')
    diagram.text(188, 157, r'$A\,(1-\omega_r/\omega_s)$', ha='center', va='center')
    diagram.text(188, 131, 'signed ledger E_cbet_iaw', fontsize=8.5, ha='center', va='center')
    diagram.text(18, 85, r'per cell: $\sum_g dQ_{c,g} + Q^{IAW}_c = 0$', fontsize=9)
    diagram.text(18, 58, r'zero detuning ($\omega_r=\omega_s$):', fontsize=9)
    diagram.text(18, 39, 'the IAW term is exactly 0', fontsize=9)
    ax = fig.add_axes([438/W, 102/H, 478/W, 256/H])
    ax.set(xlim=(G_MIN, G_MAX), ylim=(Y_MIN, Y_MAX))
    ax.plot(g, response(g), color='#2563eb', lw=1.8)
    ax.grid(color='#cbd5e1', alpha=0.5, lw=0.6)
    ax.axhline(0, color='#94a3b8', lw=0.9)
    ax.axvline(0, color='#94a3b8', lw=0.9)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.tick_params(labelsize=8.5)
    ax.set_ylabel('response $P(g)$', fontsize=10, labelpad=7)
    ax.set_xlabel('normalized detuning\n'
                  r'$g = [(\omega_q-\omega_p) - \mathbf{k}_a\cdot\mathbf{u}]\,/'
                  r'\,(|\mathbf{k}_a|\,c_a)$', fontsize=10, labelpad=14)
    ax.text(-2.42, 5.05, '$P(-g) = -P(g)$', fontsize=9)
    diagram.text(918, 375, '$g>0$: power flows from pump $q$ to probe $p$',
                 ha='right', fontsize=9)
    values = [(G0, 'black', 'operating point', (-0.60, 1.55)),
              (STRONGER, '#16a34a', 'stronger', (1.31, 4.50)),
              (WEAKER, '#f59e0b', 'weaker', (1.64, 0.70)),
              (REVERSED, '#dc2626', 'reversed: $P<0$', (-2.40, -1.15))]
    for x, color, label, label_point in values:
        ax.scatter([x], [response(x)], s=26, color=color, zorder=5)
        ax.annotate(label, xy=(x, response(x)), xytext=label_point, color=color,
                    fontsize=10, arrowprops=dict(arrowstyle='-', lw=0.7, color=color))
    for index, (x, color) in enumerate([(STRONGER, '#16a34a'), (WEAKER, '#f59e0b'),
                                        (REVERSED, '#dc2626')]):
        yy = -4.9-0.28*index
        ax.annotate('', xy=(x, yy), xytext=(G0, yy),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=0.9, mutation_scale=9,
                                    shrinkA=0, shrinkB=0))
    ax.text(0.30, -4.35, r'detuning shifts $g$ by $\Delta\omega/(|\mathbf{k}_a|c_a)$',
            ha='center', va='bottom', fontsize=9)
    ax.plot([low, high], [half, half], color='#475569', lw=1, zorder=6)
    ax.vlines([low, high], half-0.22, half+0.22, color='#475569', lw=1, zorder=6)
    ax.annotate(r'FWHM ≈ $\alpha_{iaw} = 0.2$', xy=((low+high)/2, half),
                xytext=(1.28, 3.35), fontsize=9,
                arrowprops=dict(arrowstyle='-', color='#475569', lw=0.7))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(diagram.texts)+list(ax.texts)+[ax.xaxis.label, ax.yaxis.label]
    texts += list(ax.get_xticklabels())+list(ax.get_yticklabels())
    bounds = [text.get_window_extent(renderer) for text in texts if text.get_text()]
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
