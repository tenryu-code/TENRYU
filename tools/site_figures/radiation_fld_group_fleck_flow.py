"""Multigroup FLD material linearization, implicit solve, and outer iteration.
Display 1000x340, PNG 2000x680 at dpi 200. Cv,e=rho*(de_e/dT_e)_rho;
beta=4*a_eV*T_e^3/Cv,e; f_c,g=1/(1+alpha*beta*c*dt*sigma_PA,c,g).
eta_g=c*sigma_PE,g*a_eV*(Te^n)^4*b_g(Te^n), effective scattering
(1-f_g)*c*sigma_PA,g*E_g^n. Matter absorption minus emission updates Te.
Outer tolerance 1e-5, iteration cap 20. Schematic bar uses 4 equal bands
representing bottom group 1 through top group G, not fixed physical bins.
Fonts 10/9 pt, notes 8.5; rounding 8 pixels, arrow head 12/stroke 1.3,
border 1, margin 12. Box heights fit text plus 16 pixels on centreline
y=190. Group graphic box 130x190; loop y=100; notes y=82 and 62.
All other schematic geometry uses literal display pixels.
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


W,H=1000,340
OUTPUT="docs/site/assets/radiation-fld-group-fleck-flow.png"
GROUP_COLORS=(PALE_BLUE,BLUE_FILL,PALE_BLUE,BLUE_FILL)

FLOW_Y = 190
GROUP_RECT = (16,95,130,190)
STAGE_X_WIDTHS = ((163,286),(466,325),(808,176))
BOX_PADDING = 16
LOOP_Y = 100
LOOP_NOTE_Y = 82
DEFAULT_NOTE_Y = 62


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


def box_text_layout(ax, w, title, body=(), color=INK, note=None):
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
    return rows, total


def box(ax, rect, title, body=(), fill=PALE_BLUE, edge=SLATE,
        dashed=False, color=INK, note=None):
    x, y, w, h = rect
    ax.figure._boxes.append((rect, title))
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0,rounding_size=8", facecolor=fill,
        edgecolor=edge, linewidth=1.0, linestyle="--" if dashed else "-", zorder=2))
    rows, total = box_text_layout(ax, w, title, body, color, note)
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
    # Group text and schematic together fill the compact first-stage box.
    x,y,w,h=GROUP_RECT
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=8",facecolor=PALE_BLUE,edgecolor=SLATE,lw=1))
    fig._boxes.append((GROUP_RECT,"photon-energy groups"))
    for yy,text in ((263,"photon-energy"),(245,"groups")):
        t=label(ax,81,yy,text,10,"semibold")
        fig._checks.append((t,GROUP_RECT,"photon-energy groups"))
    for i,col in enumerate(GROUP_COLORS):
        ax.add_patch(Rectangle((36,130+i*24),65,24,facecolor=col,edgecolor=SLATE,lw=.7,zorder=3))
    label(ax,68.5,142,"group 1",9)
    label(ax,68.5,214,r"group $G$",9)
    label(ax,117,130,r"$\varepsilon_0$",9)
    label(ax,117,226,r"$\varepsilon_G$",9)
    label(ax,81,110,r"$g = 1, \dots, G$",9)
    stages=[
        (r"linearize the material at $T_e^n$ (per cell)",[
            r"$C_{v,e} = \rho\,(\partial e_e/\partial T_e)_\rho$",
            r"$\beta = 4 a_{\rm eV} T_e^3 / C_{v,e}$",
            r"$f_{c,g} = 1/(1 + \alpha\beta c\Delta t\,\sigma^{PA}_{c,g})$"],PALE_BLUE,
            "constant-opacity path; table/NLTE paths use an emission-mean coefficient"),
        (r"implicit FLD solve per group for $E_g^{n+1}$",[
            r"source $f_g\eta_g$,  $\eta_g = c\,\sigma^{PE}_g a_{\rm eV} (T_e^n)^4 b_g(T_e^n)$",
            r"effective scattering $(1-f_g)\,c\,\sigma^{PA}_g E_g^n$",
            r"flux limiter $\lambda(R)$ (Levermore–Pomraning)",
            "1D: batched tridiagonal · 2D RZ: five-point CSR, CG"],BLUE_FILL,None),
        ("matter update",[
            r"absorbed $\sum_g c\,\sigma^{PA}_g E_g^{n+1}$ minus emitted $\sum_g f_g\eta_g$",r"→ new $T_e$"],GREEN_FILL,None),
    ]
    rects=[]
    for (x,w),(title,body,fill,note) in zip(STAGE_X_WIDTHS,stages):
        h=box_text_layout(ax,w,title,body,note=note)[1]+BOX_PADDING
        rect=(x,FLOW_Y-h/2,w,h)
        rects.append(rect)
        box(ax,rect,title,body,fill,note=note)
    for x1,x2 in ((146,163),(449,466),(791,808)):
        arrow(ax,[(x1,FLOW_Y),(x2,FLOW_Y)])
    source=(rects[2][0]+rects[2][2]/2,rects[2][1])
    target=(rects[0][0]+rects[0][2]/2,rects[0][1])
    arrow(ax,[source,(source[0],LOOP_Y),(target[0],LOOP_Y),target])
    loop=r"outer iteration: refresh $b_g$, opacities, $f$, $D_g$ from the new $T_e$; repeat until $\max|\Delta T_e|/T_e <$ outer_tol ($10^{-5}$) or 20 iterations"
    label(ax,500,LOOP_NOTE_Y,loop,9)
    label(ax,500,DEFAULT_NOTE_Y,"1D default: multigroup-grey synthetic acceleration; 2D RZ default: none",8.5)
    print(f"CHECK material title lines: {len(wrap(ax,stages[0][0],STAGE_X_WIDTHS[0][1]-16,10,'semibold'))}; required = 1: {'PASS' if len(wrap(ax,stages[0][0],STAGE_X_WIDTHS[0][1]-16,10,'semibold')) == 1 else 'FAIL'}")
    if len(wrap(ax,stages[0][0],STAGE_X_WIDTHS[0][1]-16,10,'semibold')) != 1:
        raise SystemExit(1)
    finish(fig,OUTPUT)

if __name__ == "__main__":
    main()
