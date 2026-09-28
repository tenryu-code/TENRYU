"""RZ FLD face stencil and five-slot CSR matrix row.

c=i*n_z+j, row=g*n_cells+c. Slots: row, row-n_z, row+n_z,
row-1, row+1; missing neighbors use (row,0). Each internal face adds
C_f=dt*A_f*D_fg/d_f to the diagonal and -C_f off diagonal.
Diagonal starts at V_c*(1+c_light*dt*sigma_PA); outer faces can add
boundary leakage. Radial area 2*pi*r_f*dz; axial area pi*(r_+^2-r_-^2).
Pattern uses n_r=4, n_z=5, highlighted (i,j)=(2,2), all internal
face coefficients nonzero. Expected 18 padded slots, 82 nonzeros,
exact symmetry. Display 900x480, DPI 200, schematic cell size 72.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
import numpy as np
from PIL import Image

W, H, DPI = 900, 480, 200
NR, NZ, HIGHLIGHT = 4, 5, (2, 2)
CELL, GRID_X, GRID_Y = 72, 65, 163
BLUE, LINE = '#2563eb', '#475569'
OUTPUT = Path('docs/site/assets/radiation-fld-rz-five-point.png')
warnings.filterwarnings('error', message='Glyph .* missing')
plt.rcParams.update({'font.family': ['Hiragino Sans', 'DejaVu Sans'],
    'font.size': 10, 'mathtext.fontset': 'stix', 'axes.linewidth': 0.9,
    'axes.edgecolor': LINE, 'axes.labelcolor': '#1e293b', 'text.color': '#1e293b',
    'xtick.color': LINE, 'ytick.color': LINE,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def check(label, value, passed):
    print(f'{label}: {value} {"PASS" if passed else "FAIL"}')
    if not passed:
        raise SystemExit(1)


def main():
    pattern = np.eye(NR*NZ, dtype=bool)
    padded = 0
    for i in range(NR):
        for j in range(NZ):
            row = i*NZ+j
            for ni, nj in ((i-1,j), (i+1,j), (i,j-1), (i,j+1)):
                if 0 <= ni < NR and 0 <= nj < NZ:
                    pattern[row, ni*NZ+nj] = True
                else:
                    padded += 1
    check('padded slots = boundary faces', padded, padded == 18)
    check('nonzeros = 5*20-18', int(pattern.sum()), pattern.sum() == 82)
    check('symmetric pattern', np.array_equal(pattern, pattern.T), np.array_equal(pattern, pattern.T))

    fig = plt.figure(figsize=(W/100, H/100), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0,W), ylim=(0,H))
    ax.axis('off')

    def text(x, y, s, **kw):
        return ax.text(x, y, s, va='center', **kw)

    def arrow(a, b, **kw):
        opts = dict(arrowstyle='-|>', mutation_scale=12, lw=1.3,
                    color=LINE, shrinkA=0, shrinkB=0)
        opts.update(kw)
        ax.add_patch(FancyArrowPatch(a, b, **opts))

    text(205, 451, 'five-point stencil on the RZ grid', ha='center',
         fontsize=11, fontweight='semibold')
    text(655, 451, 'one matrix row (group $g$, cell $c$)', ha='center',
         fontsize=11, fontweight='semibold')
    labels = {(1,1): '$c=(i,j)$', (0,1): '$R-$', (2,1): '$R+$',
              (1,0): '$Z-$', (1,2): '$Z+$'}
    for i in range(3):
        for j in range(3):
            x, y = GRID_X+i*CELL, GRID_Y+j*CELL
            center = i == 1 and j == 1
            coupled = (i,j) in labels
            ax.add_patch(Rectangle((x,y), CELL,CELL, edgecolor='#cbd5e1', lw=0.9,
                         facecolor='#dbeafe' if center else '#eff6ff' if coupled else '#f1f5f9'))
            text(x+CELL/2,y+CELL/2+20, labels.get((i,j),'not\ncoupled'), ha='center',
                 fontsize=10 if coupled else 8.5, color='#1e293b' if coupled else '#64748b')
            ax.plot(x+CELL/2,y+CELL/2, '.', color=LINE, ms=4)
    cx, cy = GRID_X+1.5*CELL, GRID_Y+1.5*CELL
    for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
        # Offset face-normal arrows from centered cell labels.
        ox, oy = (-25,0) if dy else (0,-10)
        arrow((cx+dx*23+ox,cy+dy*23+oy),(cx+dx*49+ox,cy+dy*49+oy),
              arrowstyle='<->', color=BLUE)
    yd = cy-26
    ax.plot([cx,cx], [yd-4,cy], color=LINE, lw=.7)
    ax.plot([cx+CELL,cx+CELL], [yd-4,cy], color=LINE, lw=.7)
    arrow((cx,yd),(cx+CELL,yd), arrowstyle='<->', lw=.8)
    text(cx+CELL/2,yd-12,'$d_f$',ha='center',fontsize=8.5)
    text(292, 297, r'$A_f = 2\pi r_f\,\Delta z$', fontsize=10, rotation=90)
    text(170, 395, r'$A_f = \pi(r_+^2 - r_-^2)$', ha='center')
    arrow((47,145),(95,145))
    text(106,145,'$R$')
    arrow((47,145),(47,195))
    text(47,209,'$Z$',ha='center')
    text(204,94, 'face coefficient $C_f = \\Delta t\\,A_f D_{f,g}/d_f$,\nshared by both cells',
         ha='center', linespacing=1.6, fontsize=10)

    text(650, 417, r'row $= g\,n_{cells} + c$,  $c = i\,n_z + j$', ha='center')
    starts = [427, 600, 668, 736, 804]
    widths = [167,62,62,62,62]
    names = ['diag', '$R-$', '$R+$', '$Z-$', '$Z+$']
    cols = ['row', 'row$-n_z$', 'row$+n_z$', 'row$-1$', 'row$+1$']
    for n,(x,width,name,col) in enumerate(zip(starts,widths,names,cols)):
        ax.add_patch(FancyBboxPatch((x,350),width,47,boxstyle='round,pad=0,rounding_size=8',
                     edgecolor=LINE,lw=1,facecolor='#dbeafe' if n==0 else '#eff6ff'))
        text(x+width/2,383,name,ha='center')
        text(x+width/2,365,col,ha='center',fontsize=8.5)
        text(x+width/2,315,
             '$V_c(1+c\\Delta t\\,\\sigma^{PA})$\n$+\\sum_f C_f$' if n==0 else '$-C_f$',
             ha='center',fontsize=10)
    text(651,273,'at a domain edge the slot keeps column = row with value 0',
         fontsize=8.5,ha='center')
    side, sx, sy = 10, 550, 52
    row_high = HIGHLIGHT[0]*NZ+HIGHLIGHT[1]
    ax.add_patch(Rectangle((sx,sy+(NR*NZ-1-row_high)*side),200,side,
                           facecolor='#fef3c7',edgecolor='none'))
    for row in range(NR*NZ):
        for col in range(NR*NZ):
            ax.add_patch(Rectangle((sx+col*side,sy+(19-row)*side),side,side,
                         facecolor=BLUE if pattern[row,col] else 'none',
                         edgecolor='#cbd5e1',lw=.25))
    text(651,27,r'sparsity for $n_r=4$, $n_z=5$ (row of $c=(2,2)$ highlighted)',
         ha='center',fontsize=8.5)
    fig.savefig(OUTPUT,dpi=DPI,facecolor='white')
    with Image.open(OUTPUT) as im:
        check('PNG size',im.size,im.size==(2*W,2*H))
        print(f'WROTE {OUTPUT} {im.width}x{im.height}')
    plt.close(fig)


if __name__ == '__main__':
    main()
