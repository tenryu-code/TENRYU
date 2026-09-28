"""English and Japanese reaction-product transport and energy-partition flows.
Display 960x470, PNGs 1920x940, dpi 200. Product pairs in MeV: DT (3.540,
14.049), DD Tp (1.010,3.023), DD He3n (0.820,2.449), DHe3 (3.690,14.663).
Required Q values: 17.589,4.033,3.269,18.353 MeV. Decimal arithmetic checks
exact sums (tolerance 0 MeV). Fraley alpha ion fraction is 1/(1+32/Te[keV]).
Rendering: title/body 10/9 pt, minimum 8.5; rounding 8 display pixels;
arrow head 12, stroke 1.3, border 1; margin 12. Geometry is identical in both
languages; fitted heights use the larger language text block plus 16 pixels.
Upper/lower flow row centres are y=390/89; network centre y=239.
All anchor coordinates are named constants.
"""

import os
from decimal import Decimal
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


W,H=960,470
PRODUCT_PAIRS=(("3.540","14.049"),("1.010","3.023"),("0.820","2.449"),("3.690","14.663"))
Q_VALUES=("17.589","4.033","3.269","18.353")
REACTIONS=[r"DT: D + T → $^4$He (3.540 MeV) + n (14.049 MeV)",r"DD: D + D → T (1.010 MeV) + p (3.023 MeV)",r"DD: D + D → $^3$He (0.820 MeV) + n (2.449 MeV)",r"D$^3$He: D + $^3$He → $^4$He (3.690 MeV) + p (14.663 MeV)"]
LABELS={
"en": [
("reaction network",["Bosch–Hale reactivities, RK2 subcycles"]+REACTIONS),
("charged products",[r"$^4$He, p, T, $^3$He"]),
("product treatment (scheme)",["fraley (default): point-source sphere kernel; local fraction deposited, rest to the charged-escape ledger","diffusion: multigroup slowing-down","mc: straight-line CSDA particles"]),
("electron / ion split (partition)",["li_petrasso (default, tabulated stopping)",r"fraley: $f_i = 1/(1 + 32/T_e[\mathrm{keV}])$, DT alphas only","diffusion carries its own split"]),
("neutrons",["DT-n 14.049 MeV · DD-n 2.449 MeV"]),
("neutron heating (neutron_heating=True)",["single flight, first collision with fuel D/T","mean recoil deposited locally, split e/i with the Li–Petrasso table","rest: degraded or escaped (ledgers)"]),
("electron and ion energy sources",["ledgers: deposited · escaped · in flight"])],
"ja": [
("反応ネットワーク",["Bosch–Hale の反応率、RK2 の部分ステップ"]+REACTIONS),
("荷電生成物",[r"$^4$He, p, T, $^3$He"]),
("生成物の扱い（scheme）",["fraley（既定）: 点源の球核。局所の割合を沈着し、残りは荷電粒子の逸出台帳へ","diffusion: 多群の減速","mc: 直線の CSDA 粒子"]),
("電子・イオンへの分配（partition）",["li_petrasso（既定、阻止能の表）",r"fraley: $f_i = 1/(1 + 32/T_e[\mathrm{keV}])$、DT の $\alpha$ のみ","diffusion は自身の分配を使う"]),
("中性子",["DT-n 14.049 MeV · DD-n 2.449 MeV"]),
("中性子加熱（neutron_heating=True）",["単一飛行、燃料の D/T との初回衝突","平均の反跳エネルギーを局所に沈着し、Li–Petrasso の表で電子・イオンへ分配","残り: 減速後（degraded）と逸出（escaped）の台帳へ"]),
("電子とイオンのエネルギー源",["台帳: 沈着 · 逸出 · 飛行中"])]}

# x, common row centre, width; heights fit the larger language block.
BOX_ANCHORS = ((16,239,395),(16,390,180),(212,390,390),
               (618,390,326),(16,89,284),(316,89,390),(730,89,214))
BOX_PADDING = 16


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
    for pair,q in zip(PRODUCT_PAIRS,Q_VALUES):
        value=sum(Decimal(v) for v in pair)
        error=abs(value-Decimal(q))
        ok=error==0
        print(f"CHECK Q: {pair[0]} + {pair[1]} = {value} MeV; expected {q}; error {error}; tolerance 0: {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit(1)
    fills=[PALE_BLUE,GREEN_FILL,PALE_BLUE,PALE_BLUE,AMBER_FILL,AMBER_FILL,BLUE_FILL]
    for lang,labels in LABELS.items():
        fig,ax=canvas(W,H)
        rects=[]
        for i,(x,cy,w) in enumerate(BOX_ANCHORS):
            h=max(box_text_layout(ax,w,*table[i])[1] for table in LABELS.values())+BOX_PADDING
            rects.append((x,cy-h/2,w,h))
        for rect,(title,body),fill in zip(rects,labels,fills):
            box(ax,rect,title,body,fill)
        def top(index,x):
            r=rects[index]
            return (x,r[1]+r[3])
        def bottom(index,x):
            return (x,rects[index][1])
        arrow(ax,[top(0,106),bottom(1,106)])
        arrow(ax,[(196,390),(212,390)])
        arrow(ax,[(602,390),(618,390)])
        arrow(ax,[bottom(3,837),top(6,837)])
        arrow(ax,[bottom(0,158),top(4,158)])
        arrow(ax,[(300,89),(316,89)])
        arrow(ax,[(706,89),(730,89)])
        finish(fig,f"docs/site/assets/burn-reaction-partition-{lang}.png")

if __name__ == "__main__":
    main()
