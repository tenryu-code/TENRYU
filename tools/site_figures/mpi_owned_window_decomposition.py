"""MPI owned, two-layer ghost, and unread far windows, in English/Japanese.

12 global cells 0..11; rank r owns [4*r,4*r+4). Ghosts are rank 0:
4,5; rank 1: 2,3,8,9; rank 2: 6,7. Eight owner-to-ghost copies.
Rank-1 cw=[4,8), pw/fw=[2,10). Exact checks require one copy into
each ghost and none into owned cells. Display 960x470, DPI 200.
Squares 49 display pixels, x origin 216; rank row bottoms 333,239,85.
Brackets are below rank 1 at y=218 and 176, aligned to rank-1
windows. English and Japanese share all geometry.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image

W,H,DPI=960,470,200
N_CELLS,N_RANKS,OWNED_COUNT,GHOST_LAYERS=12,3,4,2
X0,SIDE,ROW_Y=216,49,(333,239,85)
OWNED=tuple(set(range(4*r,4*r+4)) for r in range(N_RANKS))
GHOSTS=({4,5},{2,3,8,9},{6,7})
BLUE,AMBER,TEAL='#2563eb','#fbbf24','#0f766e'
LABELS={
    'en': {'index':r'global cell index $c$', 'ranks':['rank 0','rank 1','rank 2'],
           'owned':'owned','ghost':'ghost (2 layers)','far':'far (stale but finite; never read)',
           'arrow':'owner → ghost copy (exchange after owned writes)',
           'cw':'cw: owned cell\nwindow [4, 8)',
           'pw':'pw / fw: owned + ghost\nwindow [2, 10)'},
    'ja': {'index':r'大域セル番号 $c$', 'ranks':['ランク 0','ランク 1','ランク 2'],
           'owned':'所有','ghost':'ゴースト（2 層）','far':'遠方（古いが有限、読まない）',
           'arrow':'所有側 → ゴーストへ複製（所有部分の書き込み後に交換）',
           'cw':'cw: 所有セルのウィンドウ\n[4, 8)',
           'pw':'pw / fw: 所有 + ゴーストのウィンドウ\n[2, 10)'}}
warnings.filterwarnings('error',message='Glyph .* missing')
plt.rcParams.update({'font.family':['Hiragino Sans','DejaVu Sans'],'font.size':10,
    'mathtext.fontset':'stix','axes.linewidth':.9,'axes.edgecolor':'#475569',
    'axes.labelcolor':'#1e293b','text.color':'#1e293b','xtick.color':'#475569',
    'ytick.color':'#475569','figure.facecolor':'white','savefig.facecolor':'white'})


def check(label,value,passed):
    print(f'{label}: {value} {"PASS" if passed else "FAIL"}')
    if not passed:
        raise SystemExit(1)


def main():
    arrows=[(0,1,2),(0,1,3),(1,0,4),(1,0,5),
            (1,2,6),(1,2,7),(2,1,8),(2,1,9)]
    print('arrow list (owner rank, ghost rank, global cell):',arrows)
    counts={(r,c):sum(dst==r and cell==c and cell in OWNED[src]
                      for src,dst,cell in arrows) for r in range(N_RANKS) for c in sorted(GHOSTS[r])}
    check('exactly one owning-rank arrow per ghost',counts,all(n==1 for n in counts.values()))
    invalid=sum(c in OWNED[dst] for src,dst,c in arrows)
    check('arrows ending in owned cells',invalid,invalid==0)
    for lang,labels in LABELS.items():
        fig=plt.figure(figsize=(W/100,H/100),dpi=DPI)
        ax=fig.add_axes([0,0,1,1],xlim=(0,W),ylim=(0,H))
        ax.axis('off')
        def text(x,y,s,**kw):
            return ax.text(x,y,s,va='center',**kw)
        text(X0-18,410,labels['index'],ha='right')
        for c in range(N_CELLS):
            text(X0+(c+.5)*SIDE,410,str(c),ha='center')
        for r in range(N_RANKS):
            text(X0-24,ROW_Y[r]+SIDE/2,labels['ranks'][r],ha='right',fontsize=11,fontweight='semibold')
            for c in range(N_CELLS):
                owned,ghost=c in OWNED[r],c in GHOSTS[r]
                ax.add_patch(Rectangle((X0+c*SIDE,ROW_Y[r]),SIDE,SIDE,
                             fc=BLUE if owned else AMBER if ghost else '#e5e7eb',
                             ec='#cbd5e1' if owned or ghost else '#94a3b8',lw=.8,
                             hatch=None if owned or ghost else '///'))
                text(X0+(c+.5)*SIDE,ROW_Y[r]+SIDE/2,str(c),ha='center',
                     color='white' if owned else '#1e293b')
        for src,dst,c in arrows:
            x=X0+(c+.5)*SIDE
            y0=ROW_Y[src] if src<dst else ROW_Y[src]+SIDE
            y1=ROW_Y[dst]+SIDE if src<dst else ROW_Y[dst]
            ax.add_patch(FancyArrowPatch((x,y0),(x,y1),connectionstyle='arc3,rad=0.18',
                         arrowstyle='-|>',mutation_scale=12,lw=1.3,color=TEAL,shrinkA=0,shrinkB=0))
        for start,end,y,key,color in ((4,8,218,'cw',BLUE),(2,10,176,'pw','#b45309')):
            left,right=X0+start*SIDE,X0+end*SIDE
            ax.plot([left,left,right,right],[y+7,y,y,y+7],color=color,lw=1.2)
            text(left-12,y-10,labels[key],ha='right',color=color,fontsize=10,linespacing=1.4)
        ax.add_patch(FancyArrowPatch((220,446),(255,446),arrowstyle='-|>',mutation_scale=12,
                                    lw=1.3,color=TEAL,shrinkA=0,shrinkB=0))
        text(268,446,labels['arrow'],color=TEAL)
        for x,key,color,hatch in ((160,'owned',BLUE,None),(307,'ghost',AMBER,None),
                                  (538,'far','#e5e7eb','///')):
            ax.add_patch(Rectangle((x,20),16,16,fc=color,
                         ec='#94a3b8' if hatch else '#cbd5e1',hatch=hatch,lw=.7))
            text(x+24,28,labels[key])
        output=Path(f'docs/site/assets/mpi-owned-window-decomposition-{lang}.png')
        fig.savefig(output,dpi=DPI,facecolor='white')
        with Image.open(output) as im:
            check(f'{lang} PNG size',im.size,im.size==(2*W,2*H))
            print(f'WROTE {output} {im.width}x{im.height}')
        plt.close(fig)


if __name__=='__main__':
    main()
