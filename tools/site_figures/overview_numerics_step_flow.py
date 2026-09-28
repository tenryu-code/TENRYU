"""Outer time-step caps, split operators, retry, and commit for TENRYU.
Display 1040x600; output 2080x1200 at 200 dpi. Growth cap 1.2;
hydro half-steps dt/2; retry halves dt. All optional operators and the
specified radiation no-cap rule are shown. Diagram geometry uses display
pixels; text 10/9 pt, band titles 10.5 pt, notes 8.5 pt; rounded corners 8,
arrows 12-point heads and 1.3-point strokes, borders 1 point, margin >=12.
All box rectangles and routed arrow vertices are literal constants in main.
No physical computation or numerical approximation is performed.
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


W, H = 1040, 600
OUTPUT = "docs/site/assets/overview-numerics-step-flow.png"

CAP_Y, CAP_HEIGHT = 501, 57
BOX_RECTS = (
    (25,402,295,60),
    (340,402,410,60),
    (775,402,245,60),
    (775,365,245,30),
    (215,40,285,69),
    (525,40,175,69),
    (725,40,295,69),
    (600,121,420,48),
)

ARROW_ROUTES = (
    [(172,483),(172,462)],
    [(320,432),(330,432),(330,472),(897,472),(897,462)],
    [(750,432),(775,432)],
    [(897,402),(897,395)],
    [(346.5,231),(346.5,109)],
    [(793.5,231),(793.5,210),(480,210),(480,109)],
    [(500,74.5),(525,74.5)],
    [(700,74.5),(725,74.5)],
    [(740,40),(740,16),(14,16),(14,322),(212,322),(212,313)],
    [(1020,276.5),(1026,276.5),(1026,145),(1020,145)],
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
    for yy in (360,185):
        ax.plot([16,1024],[yy,yy],color=LIGHT,lw=.8,ls="--")
    for yy,s in [(575,r"1  Choose $\Delta t$ before the step"),
                 (341,"2  One outer step (execution order, left to right)"),
                 (172,"3  Accept or retry")]:
        label(ax,20,yy,s,10.5,"semibold","#334155",ha="left")
    caps=[(20,120,"hydro","acoustic CFL"),
          (152,245,"electron conduction","STS stage limit (implicit: none)"),
          (409,115,"nuclear burn","if enabled"),
          (536,120,"hot electrons","if enabled"),
          (668,170,"Braginskii viscosity","if enabled"),
          (850,170,r"FLD / $S_N$ radiation","no cap")]
    for i,(x,w,title,sub) in enumerate(caps):
        box(ax,(x,CAP_Y,w,CAP_HEIGHT),title,[sub],GREY if i==5 else PALE_BLUE,
            color=SECONDARY if i==5 else INK)
        if i<5:
            arrow(ax,[(x+w/2,501),(x+w/2,483)])
    ax.plot([80,753],[483,483],color=SLATE,lw=1.3)
    box(ax,BOX_RECTS[0],r"$\Delta t_{phys} = \min$ of the kernel caps",fill=BLUE_FILL)
    arrow(ax,ARROW_ROUTES[0])
    box(ax,BOX_RECTS[1],"operational caps",
        [r"growth $\leq 1.2\,\Delta t^{n}$ · dt.max_s · next output time · $t_{end}-t$"])
    box(ax,BOX_RECTS[2],r"$\Delta t = \min(\Delta t_{phys},$ operational caps$)$",fill=BLUE_FILL)
    arrow(ax,ARROW_ROUTES[1])
    arrow(ax,ARROW_ROUTES[2])
    box(ax,BOX_RECTS[3],r"$\Delta t <$ dt.min_s → FATAL",fill=RED_FILL,edge=RED)
    arrow(ax,ARROW_ROUTES[3],RED)
    seq=[((20,231,110,82),"entry snapshot",["(full-step retry enabled)"],True),
         ((142,231,140,82),r"$\mathcal{L}(\Delta t)$",["laser ray trace + deposition"],False),
         ((294,231,105,82),r"$\mathcal{H}(\Delta t/2)$",["hydro half-step"],False),
         ((411,231,130,82),r"$\mathcal{C}(\Delta t)$",["electron conduction"],False),
         ((553,231,176,82),r"$\mathcal{R}(\Delta t)$",[r"radiation (FLD / $S_N$) + matter exchange"],False),
         ((741,231,105,82),r"$\mathcal{H}(\Delta t/2)$",["hydro half-step"],False),
         ((858,231,162,82),"ALE rezone / remap",["2D RZ, if triggered"],True)]
    for i,(rect,title,body,dash) in enumerate(seq):
        box(ax,rect,title,body,GREY if dash else BLUE_FILL,dashed=dash)
        if i:
            prev=seq[i-1][0]
            arrow(ax,[(prev[0]+prev[2],276.5),(rect[0],276.5)])
    label(ax,1020,341,r"EOS reclosure after every operator that changes $T_e$ or $e_e$",8.5,color=SECONDARY,ha="right")
    box(ax,BOX_RECTS[4],"recoverable hydro corrector failure",["(full-step retry enabled)"],RED_FILL,RED)
    box(ax,BOX_RECTS[5],"restore entry snapshot",fill=RED_FILL,edge=RED)
    box(ax,BOX_RECTS[6],r"$\Delta t \leftarrow \Delta t/2$",["rerun the whole sequence"],RED_FILL,RED)
    arrow(ax,ARROW_ROUTES[4],RED,True)
    arrow(ax,ARROW_ROUTES[5],RED,True)
    arrow(ax,ARROW_ROUTES[6],RED)
    arrow(ax,ARROW_ROUTES[7],RED)
    label(ax,872,25,"attempt cap exhausted → hard failure",8.5,color=RED)
    arrow(ax,ARROW_ROUTES[8],RED)
    box(ax,BOX_RECTS[7],"accept and commit",[r"advance $t$ · update ledgers · write outputs (HDF5)"],GREEN_FILL,GREEN)
    arrow(ax,ARROW_ROUTES[9],GREEN)
    finish(fig,OUTPUT)

if __name__ == "__main__":
    main()
