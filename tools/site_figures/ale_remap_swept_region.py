"""Signed swept-region remap with old owner as donor.

Display 820x440, output 1640x880, dpi 200. Old faces [80,250,430,600,740],
new faces [80,250,505,600,740] in display pixels, unit cross-section.
Densities [1,2,3,4]; swept interval [430,505], width 75, mass 3*75=225.
F_(i+1/2)=-q_rec_(i+1)*abs(delta V); cell i gains and i+1 loses 225.
Mass tolerance 1e-12. RZ polygon volume is pi/3 times the cyclic sum of
(r_k+r_next)*(r_k*z_next-r_next*z_k). Strip y bounds [306,352] and
[185,231]; all coordinates are display pixels.
"""
from pathlib import Path
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image

W,H,DPI=820,440,200
OLD=np.array([80.,250.,430.,600.,740.])
NEW=np.array([80.,250.,505.,600.,740.])
DENSITIES=np.array([1.,2.,3.,4.])
TOL=1e-12
OUT=Path('docs/site/assets/ale-remap-swept-region.png')
INK,SLATE,SECONDARY,ORANGE='#1e293b','#475569','#64748b','#ea580c'
warnings.filterwarnings('error',message=r'Glyph .* missing')
plt.rcParams.update({'font.family':['Hiragino Sans','DejaVu Sans'],'font.size':10,
 'mathtext.fontset':'stix','axes.linewidth':.9,'axes.edgecolor':SLATE,
 'axes.labelcolor':INK,'text.color':INK,'xtick.color':SLATE,'ytick.color':SLATE,
 'figure.facecolor':'white','savefig.facecolor':'white'})


def main():
    before=DENSITIES*np.diff(OLD)
    swept=NEW[2]-OLD[2]
    transfer=DENSITIES[2]*swept
    after=before.copy(); after[1]+=transfer; after[2]-=transfer
    checks=[('total mass before',before.sum(),True),('total mass after',after.sum(),True),
            ('total mass difference, tolerance 1e-12',after.sum()-before.sum(),abs(after.sum()-before.sum())<TOL),
            ('cell i mass increase = 3*75 = 225, tolerance 1e-12',after[1]-before[1],abs(after[1]-before[1]-225)<TOL)]
    for name,value,passed in checks:
        print(f'CHECK {name}: {value:.15g} {"PASS" if passed else "FAIL"}')
    if not all(c[2] for c in checks):
        raise SystemExit(1)
    fig=plt.figure(figsize=(W/100,H/100),dpi=DPI)
    ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,W),ylim=(0,H)); ax.axis('off')
    for faces,y,title in [(OLD,306,'old mesh (after the Lagrangian step)'),(NEW,185,'new mesh (rezoned)')]:
        ax.text(80,y+56,title,fontsize=11,fontweight='semibold',va='bottom')
        for k,label in enumerate([r'$i-1$',r'$i$',r'$i+1$',r'$i+2$']):
            lo,hi=faces[k:k+2]
            box=FancyBboxPatch((lo,y),hi-lo,46,boxstyle='round,pad=0,rounding_size=8',
                              facecolor='none',edgecolor=SLATE,lw=1,zorder=3)
            ax.add_patch(box)
            fill=Rectangle((lo,y),hi-lo,46,facecolor='#eff6ff',edgecolor='none',zorder=1)
            fill.set_clip_path(box)
            ax.add_patch(fill)
            label_x=(lo+hi)/2
            if k==2 and faces is OLD:
                label_x=(NEW[2]+hi)/2
            if k==1 and faces is NEW:
                label_x=(lo+OLD[2])/2
            ax.text(label_x,y+23,label,ha='center',va='center',fontsize=10,zorder=4)
            if (faces is OLD and k==2) or (faces is NEW and k==1):
                overlay=Rectangle((OLD[2],y),NEW[2]-OLD[2],46,facecolor='#ffedd5',
                                  edgecolor=ORANGE,hatch='////',lw=0,zorder=2)
                overlay.set_clip_path(box)
                ax.add_patch(overlay)
    ax.plot([430,430],[185,231],color=SLATE,lw=.8,ls='--',zorder=4)
    ax.text(430,298,r'$f = i+\frac{1}{2}$',ha='right',va='top',fontsize=10)
    ax.add_patch(FancyArrowPatch((430,395),(505,395),arrowstyle='-|>',mutation_scale=12,
                                shrinkA=0,shrinkB=0,lw=1.3,color=SLATE))
    ax.text(467.5,411,r'face $f$ moves towards cell $i+1$',ha='center',fontsize=10)
    ax.add_patch(FancyArrowPatch((467.5,306),(355,231),arrowstyle='-|>',mutation_scale=12,
                                shrinkA=0,shrinkB=0,lw=2.5,color=ORANGE))
    ax.text(90,295,r'donor = cell $i+1$'+'\n'+r'(old owner of $P_f$)',fontsize=10,color=ORANGE,va='top')
    ax.annotate(r'$P_f$ (swept region)',xy=(490,311),xytext=(497,278),fontsize=10,
                arrowprops={'arrowstyle':'-','color':ORANGE,'lw':.8})
    ax.text(798,291,'If the face moves\n'+r'towards cell $i$, the'+'\n'+r'donor is cell $i$.',
            ha='right',va='top',fontsize=8.5,color=SECONDARY,linespacing=1.3)
    ax.text(26,145,r'one signed volume $\Delta V_f = V(P_f)$, one donor, equal-and-opposite updates:',fontsize=9)
    ax.text(26,119,r'$\bar q_i^{new} = \bar q_i^{old} - F_{i+1/2} + F_{i-1/2}$, here $F_{i+1/2} = -\,q^{rec}_{i+1}\,|\Delta V_f| < 0$:'
            +'\n'+r'cell $i$ gains what cell $i+1$ loses',fontsize=9,va='top',linespacing=1.55)
    ax.text(26,61,r'In 2D RZ, $P_f$ is the meridional polygon between the old and new face:'+'\n'+
            r'$\Delta V_f = \frac{\pi}{3}\sum_k (r_k + r_{k+1})(r_k z_{k+1} - r_{k+1} z_k)$',
            fontsize=8.5,color=SECONDARY,va='top',linespacing=1.65)
    fig.savefig(OUT,dpi=DPI,facecolor='white'); plt.close(fig)
    with Image.open(OUT) as im:
        print(f'WROTE {OUT} {im.width}x{im.height}')
        if im.size != (2*W,2*H):
            raise SystemExit('FAIL pixel dimensions')

if __name__=='__main__':
    main()
