"""Small TENRYU RZ mesh families, computed from generator node/connectivity rules.

Sources: src/mesh/mesh.cu:create_mesh (uniform rectangular and symmetric
tri_fan polar branches), polar_halfplane_point (exact poles/equator),
geometric_sum, solve_geometric_ratio, delayed_quintic_blend,
nearest_theta_index, polar_in_box_inset_point, build_polar_in_box_collar_layout,
and polar_in_box_exterior_ray (baseline branch). cone_theta_wall is unset, so
apply_polar_in_box_cone_postpass returns immediately; polar_in_box_q_rect is
only needed by the unused cone override. src/mesh/pentagon_belt_shell.cu:
create_pentagon_belt_shell_mesh supplies the nested angular ladder, annular
radial ladder, and five-vertex belt connectivity. The reference deck is
examples/verification/2d_rz_pentagon_belt_killer_belt_48x128.py (kappa=4).

Parameters: rectangular R=[0,1], Z=[-1,1], nr=6, nz=12, uniform. Polar:
s=[0,1], nr=6, ntheta=12, theta=j*pi/12, tri_fan center (all 13 center
node IDs retained), equal-mu off, theta_min=0, center_z=0. Coordinates are
(s*sin(theta), s*cos(theta)), with exact poles/equator and southern mirroring.
Polar-in-box: box R=[0,1], Z=[-1,1], center_z=0, ntheta=12, explicit prefix
s=[0,0.25], prefix_nr=1, morph_rings=2, collar_rings=2, morph_growth_max=3,
tri_fan center, no cone. These are the smallest counts with a nonempty
prefix, an intermediate morph ring, and two collar layers. Corners are
nearest theta indices (3,9). Morph radii are s_b+delta_b*sum(q**l,l=0..k-1);
angles use delayed quintic u**3*(10-15*u+6*u*u), u=clip((k/nm-0.2)/0.8).
The collar inset solves inset=median(exit thickness)*sum(qc**l,l=0..nc-1),
qc=min(growth_max,1.10), using the source's 256 samples and 96 bisections.
Pentagon belt: annular center as required by its generator, s_max=1, nr=5,
kappa=4, ds=1/(nr+kappa)=1/9, s_i=(i+kappa)*ds; belt_layers=[2]. Rings
0..2 have N=6 sectors, rings 3..5 have 12, with two ordinary layers on each
side. Fine theta=j*pi/12, coarse theta is its stride-2 subsample. Belt
vertices are inner[m], outer[2m], outer[2m+1], outer[2m+2], inner[m+1].
Polar/belt generator winding is clockwise in (R,Z); reverse its vertex
order for positive signed plotting areas, preserving all IDs and edges.

Display: 1000x340 at 200 dpi (2000x680 PNG), four 226x226 display-pixel
axes at x=12+250*p, y=65, shared limits R=[-0.55,1.55], Z=[-1.05,1.05],
equal aspect. Titles y=316 (11 pt), captions y=45 (9 pt), axis label
(-0.08,0) (10 pt). Cell edges #334155/0.7 pt, fills #eff6ff/#bfdbfe;
axis #94a3b8/0.9 pt dash-dot, spanning Z=[-1.04,1.04]. Checks: area>0,
R>=-1e-12, axis R exactly zero, six pentagons, polar error<=1e-12,
PNG size exact, text margin>=12 display pixels, missing glyphs forbidden.
"""

import math
from pathlib import Path
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from PIL import Image

WIDTH, HEIGHT, DPI = 1000, 340, 200
RECT_NR, RECT_NZ = 6, 12
POLAR_NR, NTHETA, S_MAX = 6, 12, 1.0
R_MIN, R_MAX, Z_MIN, Z_MAX, CENTER_Z = 0.0, 1.0, -1.0, 1.0, 0.0
PREFIX_S = (0.0, 0.25)
MORPH_RINGS, COLLAR_RINGS, MORPH_GROWTH_MAX = 2, 2, 3.0
BELT_NR, BELT_LAYER, COARSE_N, KAPPA = 5, 2, 6, 4.0
BISECTIONS, BRACKET_SAMPLES = 96, 256
BLEND_DELAY, COLLAR_GROWTH_CAP = 0.20, 1.10
COORD_TOL, RATIO_TOL, RESIDUAL_TOL = 1e-12, 1e-14, 1e-14
AXES_SIZE, AXES_BOTTOM, AXES_LEFT, PANEL_STEP = 226, 65, 12, 250
TITLE_Y, CAPTION_Y, MARGIN = 316, 45, 12
OUTPUT = Path("docs/site/assets/mesh-2d-family-gallery.png")
TITLES = ("rectangular_rz", "structured polar", "polar-in-box", "pentagon belt (CSR)")
CAPTIONS = (
    r"implicit $(i, j)$ connectivity",
    "$ (R, Z) = (s\\sin\\theta,\\ s\\cos\\theta)$,\n" + r"$\theta\in[0,\pi]$",
    "polar prefix → box-aligned\ncollar rings",
    "coarse ring $N$ → fine ring $2N$\nthrough five-sided cells",
)

warnings.filterwarnings("error", message=r"Glyph .* missing.*")
matplotlib.rcParams.update({
    "font.family": ["Hiragino Sans", "DejaVu Sans"], "font.size": 10,
    "mathtext.fontset": "stix", "axes.linewidth": 0.9,
    "axes.edgecolor": "#475569", "axes.labelcolor": "#1e293b",
    "text.color": "#1e293b", "xtick.color": "#475569",
    "ytick.color": "#475569", "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def check(label, value, condition, requirement):
    print(f"CHECK {label}: {value} {'PASS' if condition else 'FAIL'} ({requirement})")
    if not condition:
        raise RuntimeError(label)


def polar_ring(radius, theta):
    n = len(theta) - 1
    points = np.zeros((n + 1, 2))
    for j in range(n // 2 + 1):
        points[j] = (0.0 if j == 0 else radius * math.sin(theta[j]),
                     0.0 if 2 * j == n else radius * math.cos(theta[j]))
    for j in range(n // 2 + 1, n + 1):
        points[j] = (points[n - j, 0], -points[n - j, 1])
    return points


def structured_cells(nr, nz, polar=False):
    cells = []
    for i in range(nr):
        for j in range(nz):
            a, b = i * (nz + 1) + j, (i + 1) * (nz + 1) + j
            cell = [a, b, b + 1, a + 1]
            cells.append(cell[::-1] if polar else cell)
    return cells


def geometric_sum(q, count):
    if count <= 0:
        return 0.0
    if abs(q - 1.0) <= COORD_TOL:
        return float(count)
    total, term = 0.0, 1.0
    for _ in range(count):
        total += term
        term *= q
    return total


def solve_ratio(clearance, first_width, count):
    if clearance < first_width or count <= 1:
        raise RuntimeError("Invalid morph clearance or ring count")
    def residual(q):
        return first_width * geometric_sum(q, count) - clearance
    if abs(residual(1.0)) <= RESIDUAL_TOL * max(clearance, first_width):
        return 1.0
    lo, hi = 0.0, 1.0
    if residual(1.0) < 0.0:
        lo, hi = 1.0, 2.0
        while residual(hi) < 0.0:
            hi *= 2.0
            if not math.isfinite(hi) or hi >= 1e6:
                raise RuntimeError("Geometric-ratio bracket overflow")
    for _ in range(BISECTIONS):
        mid = 0.5 * (lo + hi)
        if residual(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def blend(xi):
    u = min(1.0, max(0.0, (xi - BLEND_DELAY) / (1.0 - BLEND_DELAY)))
    return u**3 * (10.0 + u * (-15.0 + 6.0 * u))


def inset_point(theta, jtr, jbr, j, inset):
    r, top, bottom = R_MAX - inset, Z_MAX - inset, Z_MIN + inset
    if j == jtr:
        return np.array([r, top])
    if j == jbr:
        return np.array([r, bottom])
    if j < jtr:
        u = (theta[j] - theta[0]) / (theta[jtr] - theta[0])
        return np.array([u * r, top])
    if j < jbr:
        u = (theta[j] - theta[jtr]) / (theta[jbr] - theta[jtr])
        return np.array([r, top - u * (top - bottom)])
    u = (theta[j] - theta[jbr]) / (theta[-1] - theta[jbr])
    return np.array([(1.0 - u) * r, bottom])


def polar_in_box(theta):
    sb, db = PREFIX_S[-1], PREFIX_S[-1] - PREFIX_S[-2]
    jtr = int(np.argmin(abs(theta - math.atan2(R_MAX, Z_MAX - CENTER_Z))))
    jbr = int(np.argmin(abs(theta - (math.pi - math.atan2(R_MAX, CENTER_Z - Z_MIN)))))
    qc = min(MORPH_GROWTH_MAX, COLLAR_GROWTH_CAP)
    factor = geometric_sum(qc, COLLAR_RINGS)

    def evaluate(inset):
        targets, ratios, thicknesses = [], [], []
        for j, angle in enumerate(theta):
            target = inset_point(theta, jtr, jbr, j, inset)
            phi_target = math.atan2(target[0], target[1] - CENTER_Z)
            q = solve_ratio(math.hypot(*target) - sb, db, MORPH_RINGS)
            phi = angle + blend((MORPH_RINGS - 1) / MORPH_RINGS) * (phi_target - angle)
            radius = sb + db * geometric_sum(q, MORPH_RINGS - 1)
            previous = (0.0 if j in (0, NTHETA) else radius * math.sin(phi),
                        CENTER_Z + radius * math.cos(phi))
            thickness = (target[1] - previous[1] if j < jtr else
                         target[0] - previous[0] if j <= jbr else
                         previous[1] - target[1])
            if not math.isfinite(thickness) or thickness <= 0.0:
                raise RuntimeError("Nonpositive morph exit thickness")
            targets.append(target)
            ratios.append(q)
            thicknesses.append(thickness)
        return float(np.median(thicknesses)), np.array(ratios), np.array(targets)

    max_inset = min(R_MAX, Z_MAX - CENTER_Z, CENTER_Z - Z_MIN) - sb - db
    lo = 0.0
    for sample in range(1, BRACKET_SAMPLES + 1):
        hi = max_inset * sample / (BRACKET_SAMPLES + 1)
        if hi - factor * evaluate(hi)[0] >= 0.0:
            break
        lo = hi
    else:
        raise RuntimeError("Collar inset could not be bracketed")
    for _ in range(BISECTIONS):
        mid = 0.5 * (lo + hi)
        if mid - factor * evaluate(mid)[0] < 0.0:
            lo = mid
        else:
            hi = mid
    inset = 0.5 * (lo + hi)
    first_width, ratios, targets = evaluate(inset)
    check("polar-in-box max morph ratio", f"{max(ratios):.16g}",
          max(ratios) <= MORPH_GROWTH_MAX * (1 + RATIO_TOL), "<=3*(1+1e-14)")
    print(f"INFO polar-in-box inset={inset:.16g} first_width={first_width:.16g} corners=({jtr},{jbr})")
    rings = [polar_ring(s, theta) for s in PREFIX_S]
    for k in range(1, MORPH_RINGS + 1):
        if k == MORPH_RINGS:
            rings.append(targets)
            continue
        ring = []
        for j, angle in enumerate(theta):
            phi = angle + blend(k / MORPH_RINGS) * (math.atan2(targets[j, 0], targets[j, 1]) - angle)
            radius = sb + db * geometric_sum(ratios[j], k)
            ring.append((0.0 if j in (0, NTHETA) else radius * math.sin(phi),
                         CENTER_Z + radius * math.cos(phi)))
        rings.append(np.array(ring))
    scale = inset / (first_width * factor)
    for k in range(1, COLLAR_RINGS + 1):
        remaining = COLLAR_RINGS - k
        offset = 0.0 if remaining == 0 else scale * first_width * qc**k * geometric_sum(qc, remaining)
        rings.append(np.array([inset_point(theta, jtr, jbr, j, offset) for j in range(NTHETA + 1)]))
    return np.concatenate(rings), structured_cells(len(rings) - 1, NTHETA, polar=True)


def pentagon_belt(theta):
    rings, offsets, counts = [], [], []
    ds = S_MAX / (BELT_NR + KAPPA)
    for i in range(BELT_NR + 1):
        n = COARSE_N if i <= BELT_LAYER else 2 * COARSE_N
        offsets.append(sum(len(ring) for ring in rings))
        counts.append(n)
        rings.append(polar_ring((i + KAPPA) * ds, theta[::NTHETA // n]))
    cells = []
    for i in range(BELT_NR):
        for j in range(counts[i]):
            a = offsets[i] + j
            if i == BELT_LAYER:
                b = offsets[i + 1] + 2 * j
                cell = [a, b, b + 1, b + 2, a + 1]
            else:
                b = offsets[i + 1] + j
                cell = [a, b, b + 1, a + 1]
            cells.append(cell[::-1])
    return np.concatenate(rings), cells


def main():
    theta = np.arange(NTHETA + 1) * math.pi / NTHETA
    radial = np.arange(POLAR_NR + 1) * S_MAX / POLAR_NR
    rectangle = np.array([(r, z) for r in np.linspace(R_MIN, R_MAX, RECT_NR + 1)
                          for z in np.linspace(Z_MIN, Z_MAX, RECT_NZ + 1)])
    polar = np.concatenate([polar_ring(s, theta) for s in radial])
    meshes = [(rectangle, structured_cells(RECT_NR, RECT_NZ)),
              (polar, structured_cells(POLAR_NR, NTHETA, polar=True)),
              polar_in_box(theta), pentagon_belt(theta)]
    reference = np.array([(s * math.sin(t), s * math.cos(t)) for s in radial for t in theta])
    error = float(np.max(abs(polar - reference)))
    check("structured polar mapping max error", f"{error:.16g}", error <= COORD_TOL, "<=1e-12")

    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=DPI)
    texts = []
    for p, (nodes, cells) in enumerate(meshes):
        polygons = [nodes[cell] for cell in cells]
        areas = np.array([0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1)
                                       - poly[:, 1] * np.roll(poly[:, 0], -1)) for poly in polygons])
        check(f"panel {p + 1} minimum signed area", f"{areas.min():.16g}", bool(np.all(areas > 0)), ">0, CCW")
        check(f"panel {p + 1} minimum R", f"{nodes[:, 0].min():.16g}", bool(np.all(nodes[:, 0] >= -COORD_TOL)), ">=-1e-12")
        axis_nodes = nodes[abs(nodes[:, 0]) <= COORD_TOL, 0]
        check(f"panel {p + 1} axis node count", len(axis_nodes), bool(len(axis_nodes) and np.all(axis_nodes == 0.0)), "all axis R exactly 0; includes coincident center IDs")
        if p == 3:
            pentagons = sum(len(cell) == 5 for cell in cells)
            check("panel 4 five-sided cells", pentagons, pentagons == COARSE_N, "=6 splitting coarse sectors")
        ax = fig.add_axes([(AXES_LEFT + PANEL_STEP * p) / WIDTH,
                           AXES_BOTTOM / HEIGHT, AXES_SIZE / WIDTH, AXES_SIZE / HEIGHT])
        ax.set(xlim=(-0.55, 1.55), ylim=(-1.05, 1.05), aspect="equal")
        ax.axis("off")
        ax.add_collection(PolyCollection(polygons, facecolors=["#bfdbfe" if p == 3 and len(cell) == 5 else "#eff6ff" for cell in cells],
                                         edgecolors="#334155", linewidths=0.7))
        ax.plot([0, 0], [-1.04, 1.04], color="#94a3b8", linestyle="-.", linewidth=0.9)
        if p == 0:
            texts.append(ax.text(-0.08, 0.0, "$R=0$", ha="right", va="center", fontsize=10))
        center = (AXES_LEFT + PANEL_STEP * p + AXES_SIZE / 2) / WIDTH
        texts.append(fig.text(center, TITLE_Y / HEIGHT, TITLES[p], ha="center", va="center", fontsize=11, fontweight="semibold"))
        texts.append(fig.text(center, CAPTION_Y / HEIGHT, CAPTIONS[p], ha="center", va="top", fontsize=9, linespacing=1.25))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in texts]
    margin = min(min(b.x0, b.y0, WIDTH * 2 - b.x1, HEIGHT * 2 - b.y1) / 2 for b in boxes)
    check("minimum text canvas margin (display px)", f"{margin:.6f}", margin >= MARGIN, ">=12")
    fig.savefig(OUTPUT, dpi=DPI, facecolor="white")
    plt.close(fig)
    with Image.open(OUTPUT) as png:
        check("PNG dimensions", f"{png.width}x{png.height}", png.size == (2 * WIDTH, 2 * HEIGHT), "=2000x680")
        print(f"WROTE {OUTPUT} {png.width}x{png.height}")


if __name__ == "__main__":
    main()
