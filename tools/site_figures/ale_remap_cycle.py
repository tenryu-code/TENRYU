"""ALE trigger, rezone admissibility, conservative remap, and rejection cycle.
Display 1080x380, PNG 2160x760, dpi 200. Trigger cadence 5, quality threshold
q_c=J_min/J_max<0.2, blend lambda=1,1/2,...,1/32, 2x2 Gauss-J gates.
No numerical model is evaluated. Shared rendering parameters: fonts 10/9 pt,
8.5 pt minimum; 8-pixel rounded corners; arrow head 12, stroke 1.3, border 1;
12-pixel canvas margin. Top bar spans y=18..58; main row y=90..220;
second row y=260..350, measured from the top. Route x=1066 returns
from the bar to next step; yes-branch horizontal is at top-origin y=235.
Literal rectangles and vertices specify all geometry.
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


W,H=1080,380
OUTPUT="docs/site/assets/ale-remap-cycle.png"

# Rectangles use bottom-origin coordinates; requested positions use top-origin.
BOX_RECTS = (
    (14,160,113,130),
    (143,160,207,130),
    (366,179,110,92),
    (492,160,167,130),
    (675,160,237,130),
    (928,173,128,104),
    (390,322,665,40),
    (350,30,330,90),
    (700,42.5,210,65),
    (930,50,120,50),
)
ARROW_ROUTES = (
    [(127,225),(143,225)],
    [(350,225),(366,225)],
    [(476,225),(492,225)],
    [(659,225),(675,225)],
    [(912,225),(928,225)],
    [(421,271),(421,322)],
    [(992,277),(992,322)],
    [(992,173),(992,145),(515,145),(515,120)],
    [(680,75),(700,75)],
    [(910,75),(930,75)],
    [(1055,342),(1066,342),(1066,75),(1050,75)],
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
    lines = wrap(ax, title, w*.80, 10, "semibold")
    for i, line in enumerate(lines):
        label(ax,x+w/2,y+h/2+8*(len(lines)-1)-16*i,line,10,"semibold")


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
    crossings = 0
    for i, route in enumerate(fig._routes):
        for other in fig._routes[i+1:]:
            for a,b in zip(route,route[1:]):
                for c,d in zip(other,other[1:]):
                    if a[0] == b[0] and c[1] == d[1]:
                        hit = min(c[0],d[0]) <= a[0] <= max(c[0],d[0]) and min(a[1],b[1]) <= c[1] <= max(a[1],b[1])
                    elif a[1] == b[1] and c[0] == d[0]:
                        hit = min(a[0],b[0]) <= c[0] <= max(a[0],b[0]) and min(c[1],d[1]) <= a[1] <= max(c[1],d[1])
                    elif a[0] == b[0] == c[0] == d[0]:
                        hit = max(min(a[1],b[1]),min(c[1],d[1])) <= min(max(a[1],b[1]),max(c[1],d[1]))
                    elif a[1] == b[1] == c[1] == d[1]:
                        hit = max(min(a[0],b[0]),min(c[0],d[0])) <= min(max(a[0],b[0]),max(c[0],d[0]))
                    else:
                        hit = False
                    crossings += bool(hit)
    print(f"CHECK route crossings: {crossings}; tolerance = 0: {'PASS' if crossings == 0 else 'FAIL'}")
    if crossings:
        failures.append("arrows cross other arrows")
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
    box(ax,BOX_RECTS[0],"after the Lagrangian step",fill=GREY)
    box(ax,BOX_RECTS[1],"trigger check",["cadence every_n_steps (5)",r"cell quality $q_c = J_{min}/J_{max} < 0.2$","axis-margin guard · corner-J trigger (opt-in)"])
    diamond(ax,BOX_RECTS[2],"triggered?")
    box(ax,BOX_RECTS[3],"rezone",["target node coordinates only","(Winslow family, reference/barrier, m1_tmop)"])
    box(ax,BOX_RECTS[4],"admissibility gates",[r"blend $\lambda$ = 1, 1/2, …, 1/32","RZ volume · corner-J · 2×2 Gauss-J · axis margin · node path · positive remap volumes"])
    diamond(ax,BOX_RECTS[5],r"some $\lambda$ passes?")
    box(ax,BOX_RECTS[6],"stay Lagrangian: mesh, state and corner masses unchanged",fill=GREY)
    box(ax,BOX_RECTS[7],"conservative remap",["swept volumes, one donor per face","equal-and-opposite updates of mass, momentum, energies, radiation-group energy"],BLUE_FILL)
    box(ax,BOX_RECTS[8],"rebuild corner masses",["accepted remap only"],BLUE_FILL)
    box(ax,BOX_RECTS[9],"next step",fill=GREEN_FILL)
    for points in ARROW_ROUTES:
        arrow(ax,points)
    label(ax,484,249,"yes",8.5)
    label(ax,437,300,"no",8.5)
    label(ax,978,300,"no: reject",8.5,ha="right")
    label(ax,1008,158,"yes",8.5)
    finish(fig,OUTPUT)

if __name__ == "__main__":
    main()
