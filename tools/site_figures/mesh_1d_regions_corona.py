"""GXII region painting and density on a broken radial axis.

Display 940x440, output 1880x880, dpi 200. DT rho=0.010 for r<=230 um;
CH rho=1.05 for 230<r<=250; corona max(0.05 exp(-(r-250)/2), 3e-4)
for 250<r<=260; exterior masked. Critical rho=0.028 at wavelength 351 nm,
rc=250+2 ln(0.05/0.028). Radial intervals [0,200], [222,264], widths
24%,72%, gap 4%; density limits [1e-4,3]; monotonic grid step 0.01 um.
"""
from pathlib import Path
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from PIL import Image

W, H, DPI = 940, 440, 200
R_IN, R, R_END = 230.0, 250.0, 260.0
RHO_DT, RHO_CH, RHO0, L, GUARD, CRITICAL = .010, 1.05, .05, 2., 3e-4, .028
GRID_STEP = .01
OUT = Path('docs/site/assets/mesh-1d-regions-corona.png')
INK, SLATE, SECONDARY, BLUE, RED = '#1e293b', '#475569', '#64748b', '#2563eb', '#dc2626'
warnings.filterwarnings('error', message=r'Glyph .* missing')
plt.rcParams.update({'font.family': ['Hiragino Sans', 'DejaVu Sans'], 'font.size': 10,
 'mathtext.fontset': 'stix', 'axes.linewidth': .9, 'axes.edgecolor': SLATE,
 'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': SLATE, 'ytick.color': SLATE,
 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def unit_label(ax, prefix, xy, text_xy=None, color=INK, coordinates='data', alignment=(0,0)):
    # Keep the Unicode unit outside the mathtext parser so mu stays upright.
    label = HPacker(children=[TextArea(prefix, textprops={'fontsize':10, 'color':color}),
                              TextArea('μm', textprops={'fontsize':10, 'color':color,
                                                        'parse_math':False, 'fontfamily':'DejaVu Sans', 'fontstyle':'normal'})],
                    align='baseline', pad=0, sep=0)
    arrow = None if text_xy is None else {'arrowstyle':'-', 'color':color, 'lw':.8}
    artist = AnnotationBbox(label, xy, xybox=text_xy, xycoords=coordinates,
                           boxcoords=coordinates, box_alignment=alignment,
                           frameon=False, pad=0, arrowprops=arrow)
    ax.add_artist(artist)


def main():
    rc = R + L*np.log(RHO0/CRITICAL)
    grid = np.linspace(R, R_END, int(round((R_END-R)/GRID_STEP))+1)
    ramp = np.maximum(RHO0*np.exp(-(grid-R)/L), GUARD)
    checks = [
        ('rho(260) > guard=0.0003', ramp[-1], ramp[-1] > GUARD),
        ('critical radius [um], inside (250,260)', rc, R < rc < R_END),
        ('rho(rc)-0.028, absolute tolerance 1e-12', RHO0*np.exp(-(rc-R)/L)-CRITICAL,
         abs(RHO0*np.exp(-(rc-R)/L)-CRITICAL) < 1e-12),
        ('max density increment on 0.01 um grid < 0', np.diff(ramp).max(), np.all(np.diff(ramp)<0))]
    for name, value, passed in checks:
        print(f'CHECK {name}: {value:.15g} {"PASS" if passed else "FAIL"}')
    if not all(c[2] for c in checks):
        raise SystemExit(1)
    fig = plt.figure(figsize=(W/100,H/100), dpi=DPI)
    left, width = 83/W, 820/W
    axes, strips = [], []
    for start, fraction, bounds in [(0,.24,(0,200)),(.28,.72,(222,264))]:
        ax = fig.add_axes([left+width*start, 61/H, width*fraction, 258/H])
        ax.set(xlim=bounds, ylim=(1e-4,3), yscale='log')
        ax.spines[['top','right']].set_visible(False)
        ax.set_xticks([0,100,200] if start == 0 else [225,230,240,250,260])
        strip = fig.add_axes([left+width*start, 348/H, width*fraction, 29/H])
        strip.set(xlim=bounds, ylim=(0,1)); strip.axis('off')
        axes.append(ax); strips.append(strip)
    a,b = axes
    a.set_ylabel(r'density $\rho$ [g/cm³]', labelpad=8)
    b.spines['left'].set_visible(False)
    b.tick_params(axis='y', which='both', left=False, labelleft=False)
    for ax in axes:
        ax.grid(axis='y', color='#cbd5e1', lw=.5, alpha=.55)
    a.plot([0,200],[RHO_DT,RHO_DT], color=BLUE, lw=2)
    b.plot([222,R_IN,R_IN,R,R],[RHO_DT,RHO_DT,RHO_CH,RHO_CH,RHO0],color=BLUE,lw=2)
    b.plot(grid,ramp,color=BLUE,lw=2)
    for ax in [b, strips[1]]:
        low, high = (1e-4,3) if ax is b else (0,1)
        ax.add_patch(Rectangle((260,low),4,high-low,facecolor='#e5e7eb',edgecolor='#94a3b8',hatch='////',lw=.6,zorder=0))
    for s in strips:
        for lo,hi,color in [(0,230,'#dbeafe'),(230,250,'#bfdbfe'),(250,260,'#e0f2fe')]:
            s.add_patch(Rectangle((lo,0),hi-lo,1,facecolor=color,edgecolor='#cbd5e1',lw=.8))
    strips[0].text(100,.5,'DT gas core',ha='center',va='center')
    strips[1].text(240,.5,'CH shell',ha='center',va='center')
    strips[1].text(255,.5,'CH corona ramp',ha='center',va='center',fontsize=10)
    strips[1].text(262,1.2,'VOID (masked)',ha='right',va='bottom',fontsize=8.5)
    fig.text(left,413/H,'region painting',fontsize=11,fontweight='semibold',va='center')
    fig.text(left,391/H,'volume fraction 1 per region',fontsize=8.5,color=SECONDARY)
    radius_label = HPacker(children=[TextArea(r'radius $r$ ', textprops={'fontsize':10}),
                                    TextArea('[μm]', textprops={'fontsize':10, 'parse_math':False, 'fontfamily':'DejaVu Sans', 'fontstyle':'normal'})],
                           align='baseline', pad=0, sep=0)
    a.add_artist(AnnotationBbox(radius_label, ((83+820/2)/W,15/H),
                               xycoords=fig.transFigure, box_alignment=(.5,0), frameon=False, pad=0))
    for ax,xx in [(a,1),(b,0)]:
        ax.plot([xx-.008,xx+.008],[-.015,.015],transform=ax.transAxes,color=SLATE,lw=1,clip_on=False)
        ax.plot([xx-.008,xx+.008],[1-.015,1+.015],transform=ax.transAxes,color=SLATE,lw=1,clip_on=False)
    for ax,xx in [(strips[0],1),(strips[1],0)]:
        ax.plot([xx-.008,xx+.008],[-.08,.08],transform=ax.transAxes,color=SLATE,lw=1,clip_on=False)
    b.axhline(CRITICAL,color=RED,ls='--',lw=1.1)
    b.text(231,.042,'critical density of CH at 351 nm\n(fully ionized): 0.028 g/cm³',
           fontsize=8.5,color=RED,va='bottom',linespacing=1.2,
           bbox={'facecolor':'white','edgecolor':'none','pad':1})
    b.plot(rc,CRITICAL,'o',color=RED,ms=4)
    unit_label(b, r'critical surface $r_c$ ≈ 251.2 ', (rc,CRITICAL), (233,.0048), color=RED)
    b.plot([250,264],[GUARD,GUARD],ls=':',color=SECONDARY,lw=1.2)
    b.annotate('tail guard 3×10⁻⁴ g/cm³ (lower bound, not a plateau)',xy=(256,GUARD),xytext=(223,1.45e-4),
        fontsize=8.5,color=SECONDARY,arrowprops={'arrowstyle':'-','color':SECONDARY,'lw':.7})
    unit_label(b, r'$\rho_0\,e^{-(r-R)/L}$,  $\rho_0 = 0.05$ g/cm³,  $L = 2$ ',
               (255,RHO0*np.exp(-2.5)), (224,.00085))
    unit_label(b, r'target surface $R$ = 250 ', (250,.05), (239,.23))
    b.text(262,.026,'VOID: masked from conduction\nand radiation coupling',rotation=90,
        fontsize=8.5,ha='center',va='center',linespacing=1.15)
    fig.savefig(OUT,dpi=DPI,facecolor='white')
    plt.close(fig)
    with Image.open(OUT) as im:
        print(f'WROTE {OUT} {im.width}x{im.height}')
        if im.size != (2*W,2*H):
            raise SystemExit('FAIL pixel dimensions')

if __name__ == '__main__':
    main()
