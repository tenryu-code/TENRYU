"""Spitzer-Harm and SNB flux addition followed by one harmonic flux cap.
Display 960x380, PNG 1920x760 at dpi 200. q_SH=-kappa_SH*grad(T_e),
delta_q=-sum_g(lambda_tr,g/3)*grad(H_g), q_raw=q_SH+delta_q;
v_th,e=sqrt(k_B*T_e/m_e), q_max=f_lim*n_e*k_B*T_e*v_th,e,
theta=1/(1+abs(q_raw)/q_max), q=theta*q_raw. Local delta_q=0.
Inset uses 401 log-spaced x samples from 0.01 to 100, y=x/(1+x),
asymptotes y=x and y=1; strict maximum y/min(x,1)<1 check.
Fonts 10/9 pt, notes 8.5, plot title 11 semibold, blue curve width 1.8;
rounding 8 pixels, arrow head 12/stroke 1.3, border 1, margin 12.
Flow centreline y=210; sum node x=228, radius 12; box rectangles and
arrow routes are named constants. Inset and footnote retain their positions.
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


W,H=960,380
OUTPUT="docs/site/assets/conduction-flux-composition.png"
X_MIN,X_MAX,N_SAMPLES=1e-2,1e2,401

FLOW_Y = 210
PLUS_CENTER = (228,FLOW_Y)
PLUS_RADIUS = 12
BOX_RECTS = (
    (16,230,188,110),
    (16,65,188,130),
    (260,132,212,156),
    (492,173,94,74),
)
ARROW_ROUTES = (
    [(204,285),(228,285),(228,222)],
    [(204,130),(228,130),(228,198)],
    [(240,210),(260,210)],
    [(472,210),(492,210)],
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
    box(ax,BOX_RECTS[0],"Spitzer–Härm flux",[r"$q_{SH} = -\kappa_{SH}\nabla T_e$",r"$\kappa_{SH}$ with the Z correction $\xi(Z)$"])
    box(ax,BOX_RECTS[1],'SNB correction (nonlocal_model = "snb")',[r"$\delta q = -\sum_g (\lambda^{tr}_g/3)\,\nabla H_g$"])
    ax.add_patch(Circle(PLUS_CENTER,PLUS_RADIUS,facecolor="white",edgecolor=SLATE,lw=1))
    label(ax,*PLUS_CENTER,"+",11)
    label(ax,338,338,r"$q_{raw} = q_{SH} + \delta q$",10)
    label(ax,338,315,r"(local: $q_{raw} = q_{SH}$)",8.5,color=SECONDARY)
    arrow(ax,ARROW_ROUTES[0])
    arrow(ax,ARROW_ROUTES[1])
    box(ax,BOX_RECTS[2],"one shared harmonic cap",[r"$q_{max} = f_{lim}\, n_e k_B T_e\, v_{th,e}$,  $v_{th,e} = \sqrt{k_B T_e/m_e}$",r"$\theta = 1/(1 + |q_{raw}|/q_{max})$"],BLUE_FILL)
    box(ax,BOX_RECTS[3],"applied flux",[r"$q = \theta\, q_{raw}$"],GREEN_FILL)
    arrow(ax,ARROW_ROUTES[2])
    arrow(ax,ARROW_ROUTES[3])
    label(ax,16,21,r'nonlocal_model = "none": $\delta q = 0$ — the same cap, no second limiter',8.5,color=SECONDARY,ha="left")
    plot=fig.add_axes([.715,.235,.26,.60])
    x=np.geomspace(X_MIN,X_MAX,N_SAMPLES)
    y=x/(1+x)
    ratio=float(np.max(y/np.minimum(x,1)))
    ok=ratio<1
    print(f"CHECK harmonic cap: max curve/min(x,1) = {ratio:.15f}; required < 1: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)
    plot.loglog(x,y,color=BLUE,lw=1.8,zorder=3)
    plot.loglog(x,x,color=SECONDARY,ls="--",lw=1,label=r"diffusive: $q \approx q_{raw}$")
    plot.axhline(1,color=SECONDARY,ls="--",lw=1,label=r"free-streaming cap $q_{max}$")
    plot.set(xlim=(X_MIN,X_MAX),ylim=(.005,150),xlabel=r"$|q_{raw}|/q_{max}$",ylabel=r"$|q|/q_{max}$")
    plot.set_title("harmonic cap",fontsize=11,fontweight="semibold",pad=10)
    plot.tick_params(labelsize=8.5)
    plot.legend(loc="upper left",fontsize=8.5,frameon=False,handlelength=1.3)
    finish(fig,OUTPUT)

if __name__ == "__main__":
    main()
