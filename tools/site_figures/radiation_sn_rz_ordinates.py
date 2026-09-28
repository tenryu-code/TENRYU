"""Product S_N quadrature and RZ wavefront launches.

n_angles=8 Gauss-Legendre mu_Z levels, n_phi_half=4,
phi=(k+1/2)*pi/4, mu_R=sqrt(1-mu_Z**2)*cos(phi), w=w_mu/4.
Launch signs are (--),(-+),(+-),(++), sorted negative mu_R first.
For n_r=5,n_z=4, stage k=a+b maps i=a or n_r-1-a and j=b or
n_z-1-b according to direction signs. Axis inflow is reflected from
negative-mu_R outgoing ordinates; outer R defaults to vacuum.
Checks: weights sum to 2 within 1e-12, projected norm squared <=1,
each launch visits all 20 cells once. Display 940x480, DPI 200.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
import numpy as np
from PIL import Image

W, H, DPI = 940, 480, 200
N_ANGLES, N_PHI_HALF, NR, NZ, WEIGHT_TOL = 8, 4, 5, 4, 1e-12
LAUNCHES = ((-1,-1),(-1,1),(1,-1),(1,1))
COLORS = ('#7c3aed','#ea580c','#16a34a','#2563eb')
OUTPUT = Path('docs/site/assets/radiation-sn-rz-ordinates.png')
warnings.filterwarnings('error', message='Glyph .* missing')
plt.rcParams.update({'font.family': ['Hiragino Sans', 'DejaVu Sans'],
    'font.size':10,'mathtext.fontset':'stix','axes.linewidth':.9,
    'axes.edgecolor':'#475569','axes.labelcolor':'#1e293b','text.color':'#1e293b',
    'xtick.color':'#475569','ytick.color':'#475569',
    'figure.facecolor':'white','savefig.facecolor':'white'})


def check(label,value,passed):
    print(f'{label}: {value} {"PASS" if passed else "FAIL"}')
    if not passed:
        raise SystemExit(1)


def main():
    levels, gl_weights = np.polynomial.legendre.leggauss(N_ANGLES)
    dirs = [(np.sqrt(1-z*z)*np.cos((p+.5)*np.pi/N_PHI_HALF),z,w/N_PHI_HALF)
            for z,w in zip(levels,gl_weights) for p in range(N_PHI_HALF)]
    dirs.sort(key=lambda d:(d[0]>=0,d[1]))
    weights = sum(d[2] for d in dirs)
    norms = np.array([r*r+z*z for r,z,w in dirs])
    check('sum w = 2, tolerance 1e-12', f'{weights:.16g}', abs(weights-2)<=WEIGHT_TOL)
    check('max(mu_R^2+mu_Z^2) <= 1', f'{norms.max():.16g}', bool(np.all(norms<=1)))
    for launch,(sr,sz) in enumerate(LAUNCHES,1):
        cells = []
        for k in range(NR+NZ-1):
            for a in range(NR):
                b=k-a
                if 0<=b<NZ:
                    cells.append((a if sr>0 else NR-1-a,b if sz>0 else NZ-1-b))
        check(f'launch {launch} unique visits',f'{len(set(cells))}/{len(cells)}',
              len(cells)==NR*NZ and set(cells)=={(i,j) for i in range(NR) for j in range(NZ)})

    fig=plt.figure(figsize=(W/100,H/100),dpi=DPI)
    ax=fig.add_axes([0,0,1,1],xlim=(0,W),ylim=(0,H))
    ax.axis('off')
    def text(x,y,s,**kw):
        return ax.text(x,y,s,va='center',**kw)
    def arrow(a,b,**kw):
        opts=dict(arrowstyle='-|>',mutation_scale=12,lw=1.3,color='#475569',shrinkA=0,shrinkB=0)
        opts.update(kw)
        ax.add_patch(FancyArrowPatch(a,b,**opts))

    text(191,451,r'ordinates projected on $(\mu_R, \mu_Z)$',ha='center',fontsize=11,fontweight='semibold')
    text(549,443,'wavefront order for launch 4\n$(\\mu_R>0,\\ \\mu_Z>0)$',
         ha='center',fontsize=11,fontweight='semibold',linespacing=1.5)
    text(820,451,'start corners',ha='center',fontsize=11,fontweight='semibold')
    cx,cy,radius=188,290,113
    ax.add_patch(Circle((cx,cy),radius,fc='none',ec='#cbd5e1',lw=1))
    arrow((cx-radius-12,cy),(cx+radius+12,cy),lw=.8)
    arrow((cx,cy-radius-12),(cx,cy+radius+12),lw=.8)
    text(cx+radius+24,cy,r'$\mu_R$')
    text(cx+12,cy+radius+8,r'$\mu_Z$')
    for k,((sr,sz),color) in enumerate(zip(LAUNCHES,COLORS),1):
        selected=[d for d in dirs if d[0]*sr>0 and d[1]*sz>0]
        ax.scatter([cx+radius*d[0] for d in selected],[cy+radius*d[1] for d in selected],
                   c=color,s=23,zorder=4)
        y=135-(k-1)*23
        ax.plot(65,y,'o',ms=5,color=color)
        text(80,y,fr'launch {k}: $\mu_R{"<" if sr<0 else ">"}0,\ \mu_Z{"<" if sz<0 else ">"}0$',fontsize=10)
    text(190,25,'$n_{angles}=8$ shown: 8 Gauss–Legendre levels × 4 azimuths',
         fontsize=8.5,ha='center')

    gx,gy,side=431,171,46
    cmap=matplotlib.colors.LinearSegmentedColormap.from_list('stages',['#eff6ff','#93c5fd'])
    for i in range(NR):
        for j in range(NZ):
            ax.add_patch(Rectangle((gx+i*side,gy+j*side),side,side,
                         fc=cmap((i+j)/(NR+NZ-2)),ec='#cbd5e1',lw=.8))
            text(gx+(i+.5)*side,gy+(j+.5)*side,str(i+j),ha='center')
    arrow((gx-25,gy+side*.5),(gx,gy+side*.5),color='#2563eb')
    arrow((gx+side*.5,gy-25),(gx+side*.5,gy),color='#2563eb')
    arrow((gx+90,gy+NZ*side+10),(gx+150,gy+NZ*side+58),color='#2563eb',lw=2.5)
    arrow((gx+NR*side-43,gy-24),(gx+NR*side,gy-24))
    text(gx+NR*side+8,gy-24,'$R$')
    arrow((gx-22,gy+NZ*side-45),(gx-22,gy+NZ*side))
    text(gx-22,gy+NZ*side+13,'$Z$',ha='center')
    text(547,99,'cells with equal $k$ are independent\nand run in parallel',ha='center',linespacing=1.6)

    for idx,((sr,sz),color) in enumerate(zip(LAUNCHES,COLORS)):
        x=735+(idx%2)*108
        y=289-(idx//2)*123
        side=18
        for i in range(3):
            for j in range(3):
                start=(i==(0 if sr>0 else 2) and j==(0 if sz>0 else 2))
                ax.add_patch(Rectangle((x+i*side,y+j*side),side,side,
                             fc=color if start else '#eff6ff',ec='#cbd5e1',lw=.7))
        text(x+27,y+72,f'launch {idx+1}',ha='center',color=color)
        start=(x+(0.5 if sr>0 else 2.5)*side,y+(0.5 if sz>0 else 2.5)*side)
        arrow(start,(start[0]+sr*36,start[1]+sz*36),color=color,lw=1.5)
        text(x-12,y+27,'axis',rotation=90,ha='center',fontsize=8.5)
        if idx==0:
            text(x+65,y+27,'outer $R$',rotation=90,ha='center',fontsize=8.5)
    text(819,90,'$\\mu_R<0$ launches run first;\nat the axis their outgoing\nintensity supplies the mirrored\n$\\mu_R>0$ inflow',
         ha='center',fontsize=8.5,linespacing=1.6)
    fig.savefig(OUTPUT,dpi=DPI,facecolor='white')
    with Image.open(OUTPUT) as im:
        check('PNG size',im.size,im.size==(2*W,2*H))
        print(f'WROTE {OUTPUT} {im.width}x{im.height}')
    plt.close(fig)


if __name__ == '__main__':
    main()
