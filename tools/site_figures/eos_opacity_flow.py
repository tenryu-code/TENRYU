"""Material and cell-state closures feeding transport and radiation coefficients.
Display 960x480, PNG 1920x960, dpi 200. n_e=Zbar*rho/(A*m_p),
Z_eff=Zbar*r2 with r2=<Z^2>/Zbar^2 clamped to [1,10]; sigma=rho*kappa.
Ideal electron pressure is Zbar*rho*k_B*T_e/(A*m_p). Planck mixing is
mass-linear and Rosseland mixing harmonic. No numerical model evaluated.
Fonts 10/9 pt, minimum 8.5; rounding 8 pixels; arrows head 12, stroke 1.3;
border 1, margin 12. All layout coordinates are literal display pixels.
"""

import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle, Rectangle
from matplotlib.path import Path as MplPath
from PIL import Image
import numpy as np

DPI = 200
FONT_SIZE = 10
BODY_SIZE = 9
MIN_SIZE = 8.5
MARGIN = 12
INK = "#1e293b"
SLATE = "#475569"
SECONDARY = "#64748b"
LIGHT = "#cbd5e1"
BLUE = "#2563eb"
PALE_BLUE = "#eff6ff"
BLUE_FILL = "#dbeafe"
GREY = "#f1f5f9"
GREEN = "#16a34a"
GREEN_FILL = "#dcfce7"
AMBER_FILL = "#fef3c7"
RED = "#dc2626"
RED_FILL = "#fee2e2"
ROUNDING = 8
ARROW_SCALE = 12
LINE_SPACING = 15
TITLE_SPACING = 17
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / "tmp/site_figures_mplconfig_C"))
warnings.filterwarnings("error", message=r"Glyph .* missing.*")
plt.rcParams.update({
    "font.family": ["Hiragino Sans", "DejaVu Sans"], "font.size": FONT_SIZE,
    "mathtext.fontset": "stix", "axes.linewidth": 0.9,
    "axes.edgecolor": SLATE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": SLATE, "ytick.color": SLATE,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


W,H=960,480
OUTPUT="docs/site/assets/eos-opacity-flow.png"

BOX_RECTS = (
    (16,355,160,105),
    (16,285,160,55),
    (195,305,245,155),
    (460,345,110,75),
    (590,305,354,155),
    (400,160,275,125),
    (705,217,239,68),
    (705,137,239,68),
    (705,25,239,100),
    (16,145,350,125),
    (16,15,350,110),
    (400,25,275,75),
)

ARROW_ROUTES = (
    [(176,407),(195,407)],
    [(176,312),(185,312),(185,337),(195,337)],
    [(440,382.5),(460,382.5)],
    [(570,382.5),(590,382.5)],
    [(625,305),(625,285)],
    [(176,375),(181,375),(181,277),(100,277),(100,270)],
    [(96,285),(96,278),(190,278),(190,270)],
    [(515,345),(515,295),(350,295),(350,270)],
    [(190,145),(190,125)],
    [(366,62.5),(400,62.5)],
)


def canvas(width, height):
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(0, height))
    ax.axis("off")
    fig._checks = []
    fig._boxes = []
    fig._routes = []
    fig._display_size = (width, height)
    fig.canvas.draw()
    return fig, ax


def label(ax, x, y, s, size=10, weight="normal", color=INK, ha="center", va="center"):
    t = ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
                ha=ha, va=va, zorder=5)
    return t


def measure(ax, s, size, weight):
    t = ax.text(0, 0, s, fontsize=size, fontweight=weight)
    extent = t.get_window_extent(ax.figure.canvas.get_renderer())
    t.remove()
    return extent.width / 2


def wrap(ax, s, width, size, weight="normal"):
    # Math expressions are indivisible; prose wraps only at existing spaces.
    words = []
    token = ""
    in_math = False
    for character in s:
        if character == "$":
            in_math = not in_math
        if character == " " and not in_math:
            if token:
                words.append(token)
                token = ""
        else:
            token += character
    if token:
        words.append(token)
    lines = []
    line = ""
    for word in words:
        candidate = line + (" " if line else "") + word
        if measure(ax, candidate, size, weight) > width and line:
            lines.append(line)
            line = word
        else:
            line = candidate
        if measure(ax, line, size, weight) > width + 0.01:
            raise ValueError(f"Text cannot fit width {width}: {line}")
    if line:
        lines.append(line)
    return lines


def box(ax, rect, title, body=(), fill=PALE_BLUE, edge=SLATE,
        dashed=False, color=INK, note=None):
    x, y, w, h = rect
    ax.figure._boxes.append((rect, title))
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0,rounding_size=8", facecolor=fill,
        edgecolor=edge, linewidth=1.0, linestyle="--" if dashed else "-", zorder=2))
    rows = [(s, 10, "semibold", color, TITLE_SPACING)
            for s in wrap(ax, title, w-16, 10, "semibold")]
    for item in body:
        rows.extend((s, 9, "normal", color, LINE_SPACING)
                    for s in wrap(ax, item, w-16, 9))
    if note:
        rows.extend((s, 8.5, "normal", SECONDARY, LINE_SPACING)
                    for s in wrap(ax, note, w-16, 8.5))
    spaced_rows = []
    for text, size, weight, text_color, spacing in rows:
        probe = ax.text(0, 0, text, fontsize=size, fontweight=weight)
        height = probe.get_window_extent(ax.figure.canvas.get_renderer()).height / 2
        probe.remove()
        spaced_rows.append((text, size, weight, text_color, max(spacing, height+3)))
    rows = spaced_rows
    total = sum(row[4] for row in rows)
    if total > h-12:
        raise ValueError(f"Text cannot fit height {h}: {title}; needs {total+12}")
    cy = y + (h+total)/2
    for s, size, weight, tc, spacing in rows:
        cy -= spacing/2
        t = label(ax, x+w/2, cy, s, size, weight, tc)
        ax.figure._checks.append((t, rect, title))
        cy -= spacing/2
    return rect


def arrow(ax, points, color=SLATE, dashed=False):
    ax.figure._routes.append(points)
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>",
        mutation_scale=ARROW_SCALE, linewidth=1.3, color=color,
        linestyle="--" if dashed else "-", shrinkA=0, shrinkB=0, zorder=3))


def diamond(ax, rect, title):
    x,y,w,h=rect
    ax.add_patch(Polygon([(x,y+h/2),(x+w/2,y+h),(x+w,y+h/2),(x+w/2,y)],
                         closed=True, facecolor=AMBER_FILL, edgecolor=SLATE, lw=1))
    label(ax,x+w/2,y+h/2,title,10,"semibold")


def finish(fig, path):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    width,height=fig._display_size
    failures=[]
    for t,rect,title in fig._checks:
        e=t.get_window_extent(renderer)
        x,y,w,h=rect
        if e.x0/2 < x+3 or e.x1/2 > x+w-3 or e.y0/2 < y+3 or e.y1/2 > y+h-3:
            failures.append(title + ": text outside box")
    for ax in fig.axes:
        for t in ax.texts:
            if not t.get_visible() or not t.get_text():
                continue
            e=t.get_window_extent(renderer)
            if e.x0/2 < MARGIN-0.05 or e.x1/2 > width-MARGIN+0.05 or e.y0/2 < MARGIN-0.05 or e.y1/2 > height-MARGIN+0.05:
                failures.append(t.get_text()+": text outside canvas margin")
    def intersects(points, bounds):
        left, bottom, right, top = bounds
        for (x1,y1),(x2,y2) in zip(points,points[1:]):
            if x1 == x2 and left < x1 < right and max(min(y1,y2),bottom) < min(max(y1,y2),top):
                return True
            if y1 == y2 and bottom < y1 < top and max(min(x1,x2),left) < min(max(x1,x2),right):
                return True
        return False
    for route in fig._routes:
        for (x,y,w,h), title in fig._boxes:
            if intersects(route, (x+.01,y+.01,x+w-.01,y+h-.01)):
                failures.append(title + ": arrow crosses box")
        for text in fig.axes[0].texts:
            e=text.get_window_extent(renderer)
            if intersects(route, (e.x0/2,e.y0/2,e.x1/2,e.y1/2)):
                failures.append(text.get_text() + ": arrow crosses text")
    print(f"CHECK layout: {len(failures)} violations; tolerance = 0: {'FAIL' if failures else 'PASS'}")
    if failures:
        raise ValueError("; ".join(failures))
    fig.savefig(path, dpi=200, facecolor="white")
    with Image.open(path) as im:
        actual=im.size
    expected=(2*width,2*height)
    ok=actual==expected
    print(f"WROTE {path} {actual[0]}x{actual[1]}")
    print(f"CHECK dimensions: {actual}, expected {expected}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)
    print("CHECK missing glyph warnings: 0; tolerance = 0: PASS")
    plt.close(fig)

def main():
    fig,ax=canvas(W,H)
    box(ax,BOX_RECTS[0],"material data",[r"$A$ (mean ion mass, amu), $Z$"],GREY)
    box(ax,BOX_RECTS[1],"cell state",[r"$\rho$, $e_e$, $e_i$"],GREY)
    box(ax,BOX_RECTS[2],"EOS closure",["ideal_gas or TMAT-H5 table",r"inverse $T(\rho, e)$: monotone binary search",r"forward $P$, $e$, $c_v$"])
    box(ax,BOX_RECTS[3],"closed state",[r"$T_e$, $T_i$, $P$, $c_v$"])
    box(ax,BOX_RECTS[4],r"ionization $\bar Z$",[r"fixed $Z$ · Thomas–Fermi (More fit) · table $\bar Z(\rho, T_e)$","re-evaluated at the conduction and radiation entries"])
    box(ax,BOX_RECTS[5],r"$n_e = \bar Z\rho/(A m_p)$",[r"$Z_{eff} = \bar Z r_2$,  $r_2 = \langle Z^2\rangle/\bar Z^2$ (clamped 1–10)"],BLUE_FILL)
    box(ax,BOX_RECTS[6],"electron conduction",[r"Spitzer $\kappa$ with $Z_{eff}$"],GREEN_FILL)
    box(ax,BOX_RECTS[7],"laser absorption",[r"inverse bremsstrahlung, $\nu_{ei}$"],GREEN_FILL)
    box(ax,BOX_RECTS[8],"electron EOS branch",[r"ideal gas: $P_e = \bar Z\rho k_B T_e/(A m_p)$"],GREEN_FILL)
    box(ax,BOX_RECTS[9],"opacity",[r"constant or table $\kappa(\rho, T)$ [cm²/g]","mixed cells: Planck mass-linear, Rosseland harmonic"],AMBER_FILL)
    box(ax,BOX_RECTS[10],"group coefficients",[r"Planck fractions $b_g(T)$",r"$\sigma = \rho\kappa$ [1/cm]: $\sigma^{PA}_g$, $\sigma^{PE}_g$, $\sigma_{R,g}$"],AMBER_FILL)
    box(ax,BOX_RECTS[11],"radiation",[r"FLD / $S_N$"],GREEN_FILL)
    arrow(ax,ARROW_ROUTES[0])
    arrow(ax,ARROW_ROUTES[1])
    arrow(ax,ARROW_ROUTES[2])
    arrow(ax,ARROW_ROUTES[3])
    arrow(ax,ARROW_ROUTES[4])
    arrow(ax,ARROW_ROUTES[5])
    arrow(ax,ARROW_ROUTES[6])
    arrow(ax,ARROW_ROUTES[7])
    for yy in (251,171,75):
        arrow(ax,[(675,222.5),(690,222.5),(690,yy),(705,yy)])
    arrow(ax,ARROW_ROUTES[8])
    arrow(ax,ARROW_ROUTES[9])
    finish(fig,OUTPUT)

if __name__ == "__main__":
    main()
