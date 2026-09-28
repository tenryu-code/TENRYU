"""Hot-electron capture, straight-chord transport, and shell deposition.

Dimensionless centre C=(0,0); fuel r<0.55, shell 0.55<r<0.70;
coronal density n=exp(-(r-0.80)/0.20), used without cutoff for ray
dynamics everywhere. Shading is restricted to r>0.70, n<=3, and
polar angles +/-38 degrees. Critical r=0.80; capture n=0.25 gives
r=0.80+0.20*ln(4). Five rays start at x=2, y=-0.45,-0.22,0,0.22,0.45,
v=(-sqrt(1-n),0). RK4 dt=0.001 integrates x'=v, v'=-grad(n)/2,
c=1. Stop at interpolated capture crossings; central ray continues
to its turning point and returns to capture, displayed with y+0.02
on the return. Continuation linewidth 1.4; a size-8 returning arrow
runs from r=0.90 to 0.94. Energy tolerance 1e-6, capture radius tolerance 1e-3.
Three middle sources emit chords at -20,-10,0,10,20 degrees about
their crossing velocity, stopping at r=0.40 or closest approach.
Chord alpha runs linearly 0.95 to 0.15; shell overlay alpha=0.35.
Radial verification arrow begins at capture angle -30 degrees,
ends at r=0.08. Spectrum exp(-u), u=E/T_h, displayed [0.1,12];
30 logarithmic groups [0.2,8], weights are differences of
(1+u)*exp(-u), normalized to one (tolerance 1e-14).
Display 900x520, DPI 200; target display origin (300,210), scale 250;
corona shown out to r=2.2, clipped to y=65..390 display pixels.
Spectrum inset (x,y,width,height)=(46,325,242,154) display pixels.
Surface labels at (448,363) and (540,399) use straight leaders from
(450,350) and (533,387) to the respective arc endpoints at 38 degrees.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Arc, Rectangle, FancyArrowPatch
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image

W,H,DPI=900,520,200
FUEL_R,SHELL_R,CRITICAL_R,DENSITY_LENGTH=0.55,0.70,0.80,0.20
CAPTURE_FRACTION=0.25
CAPTURE_R=CRITICAL_R-DENSITY_LENGTH*np.log(CAPTURE_FRACTION)
SECTOR_DEG,OUTER_R=38,2.2
START_X,OFFSETS,DT=2.0,(-.45,-.22,0.,.22,.45),1e-3
CHORD_ANGLES=(-20,-10,0,10,20)
STOP_R,RADIAL_ANGLE,RADIAL_STOP_R,RETURN_OFFSET=.40,-30,.08,.02
CONTINUATION_LW,RETURN_ARROW_SIZE,RETURN_ARROW_RADII=1.4,8,(.90,.94)
CRITICAL_LABEL_POS,CAPTURE_LABEL_POS=(448,363),(540,399)
CRITICAL_LEADER_START,CAPTURE_LEADER_START=(450,350),(533,387)
N_GROUPS,U_MIN,U_MAX=30,.2,8.
ENERGY_TOL,CAPTURE_TOL,WEIGHT_TOL=1e-6,1e-3,1e-14
ORIGIN,SCALE=np.array([300.,210.]),250.
BLUE,ORANGE,LINE='#2563eb','#ea580c','#475569'
LABELS={
 'en':{'critical':r'critical surface $n_e = n_c$',
       'capture':'capture surface $n_e = 0.25\\,n_c$\n(source_nc_fraction)',
       'continue':'$(1-\\eta)\\,P^{cross}_{ray}$ continues\n(subtract_from_laser)',
       'power':r'$P_h = \eta(t)\,P^{cross}_{ray}$',
       'cone':'cone of straight chords\n(half-angle $\\theta_{div}$)',
       'deposit':'collisional (CSDA) deposition\nalong each chord',
       'radial':'radial mode (verification): marches inward; residual → innermost cell',
       'spectrum':r'spectrum: 30 log groups on $[0.2, 8]\,T_h$',
       'fuel':'fuel','shell':'shell','corona':'corona','centre':'C'},
 'ja':{'critical':r'臨界面 $n_e = n_c$',
       'capture':'捕捉面 $n_e = 0.25\\,n_c$（source_nc_fraction）',
       'continue':'$(1-\\eta)\\,P^{cross}_{ray}$ は光線に沿って進む\n（subtract_from_laser）',
       'power':r'$P_h = \eta(t)\,P^{cross}_{ray}$',
       'cone':'直線の弦の円錐（半角 $\\theta_{div}$）',
       'deposit':'弦に沿った衝突による沈着（CSDA）',
       'radial':'radial モード（検証用）: 内向きに進み、残りは最内セルへ',
       'spectrum':r'スペクトル: $[0.2, 8]\,T_h$ に対数 30 群',
       'fuel':'燃料','shell':'シェル','corona':'コロナ','centre':'C'}}
warnings.filterwarnings('error',message='Glyph .* missing')
plt.rcParams.update({'font.family':['Hiragino Sans','DejaVu Sans'],'font.size':10,
    'mathtext.fontset':'stix','axes.linewidth':.9,'axes.edgecolor':LINE,
    'axes.labelcolor':'#1e293b','text.color':'#1e293b','xtick.color':LINE,
    'ytick.color':LINE,'figure.facecolor':'white','savefig.facecolor':'white'})


def density(position):
    return np.exp(-(np.linalg.norm(position,axis=-1)-CRITICAL_R)/DENSITY_LENGTH)


def rhs(state):
    pos=state[:2]
    acceleration=density(pos)*pos/(2*DENSITY_LENGTH*np.linalg.norm(pos))
    return np.r_[state[2:],acceleration]


def step(state):
    k1=rhs(state)
    k2=rhs(state+DT*k1/2)
    k3=rhs(state+DT*k2/2)
    k4=rhs(state+DT*k3)
    return state+DT*(k1+2*k2+2*k3+k4)/6


def check(label,value,passed):
    print(f'{label}: {value} {"PASS" if passed else "FAIL"}')
    if not passed:
        raise SystemExit(1)


def trace(offset):
    pos=np.array([START_X,offset])
    state=np.r_[pos,-np.sqrt(1-density(pos)),0.]
    states=[state.copy()]
    for _ in range(20000):
        nxt=step(state)
        if np.linalg.norm(nxt[:2])<=CAPTURE_R:
            # Interpolate the first crossing within the final RK4 step.
            p,d=state[:2],nxt[:2]-state[:2]
            roots=np.roots([np.dot(d,d),2*np.dot(p,d),np.dot(p,p)-CAPTURE_R**2])
            fraction=next(float(v.real) for v in roots if abs(v.imag)<1e-10 and 0<=v.real<=1)
            crossing=state+fraction*(nxt-state)
            states.append(crossing)
            return np.array(states),crossing
        states.append(nxt.copy())
        state=nxt
    raise RuntimeError('Ray failed to reach capture surface')


def main():
    rays=[]
    for offset in OFFSETS:
        states,crossing=trace(offset)
        err=np.max(np.abs(np.sum(states[:,2:]**2,axis=1)+density(states[:,:2])-1))
        radius_error=abs(np.linalg.norm(crossing[:2])-CAPTURE_R)
        check(f'ray y={offset:g}, max invariant error < 1e-6',f'{err:.12g}',err<ENERGY_TOL)
        check(f'ray y={offset:g}, capture radius error <= 1e-3',f'{radius_error:.12g}',radius_error<=CAPTURE_TOL)
        rays.append((states,crossing))
    # Continue with the same RK4 dynamics through the central turning point.
    state=rays[2][1].copy()
    continuation=[state.copy()]
    for _ in range(20000):
        state=step(state)
        continuation.append(state.copy())
        if state[2]>0 and state[0]>=CAPTURE_R:
            break
    else:
        raise RuntimeError('Central ray failed to return from critical surface')
    continuation=np.array(continuation)
    err=np.max(np.abs(np.sum(continuation[:,2:]**2,axis=1)+density(continuation[:,:2])-1))
    check('central continuation max invariant error < 1e-6',f'{err:.12g}',err<ENERGY_TOL)
    edges=np.geomspace(U_MIN,U_MAX,N_GROUPS+1)
    primitive_tail=(1+edges)*np.exp(-edges)
    weights=-np.diff(primitive_tail)
    fraction=weights.sum()
    weights/=fraction
    print(f'exponential spectrum in-window fraction before renormalisation: {fraction:.16g}')
    check('renormalised sum of 30 group weights, tolerance 1e-14',
          f'{weights.sum():.16g}',abs(weights.sum()-1)<=WEIGHT_TOL)
    print(f'capture radius: {CAPTURE_R:.16g}')

    for lang,labels in LABELS.items():
        fig=plt.figure(figsize=(W/100,H/100),dpi=DPI)
        ax=fig.add_axes([0,0,1,1],xlim=(0,W),ylim=(0,H))
        ax.axis('off')
        def text(x,y,s,**kw):
            return ax.text(x,y,s,va='center',**kw)
        def xy(position):
            return ORIGIN+SCALE*np.asarray(position)
        def polar(r,angle):
            a=np.deg2rad(angle)
            return xy([r*np.cos(a),r*np.sin(a)])
        def arrow(a,b,**kw):
            options=dict(arrowstyle='-|>',mutation_scale=12,lw=1.3,
                         color=LINE,shrinkA=0,shrinkB=0)
            options.update(kw)
            ax.add_patch(FancyArrowPatch(a,b,**options))
        clip=Rectangle((285,65),580,325,transform=ax.transData)
        # Density-dependent blue corona, clipped to the sector and viewport.
        xx=np.linspace(0,OUTER_R,900)
        yy=np.linspace(-.6,.6,520)
        X,Y=np.meshgrid(xx,yy)
        radius=np.hypot(X,Y)
        n=np.exp(-(radius-CRITICAL_R)/DENSITY_LENGTH)
        mask=(radius>SHELL_R)&(n<=3)&(np.abs(np.arctan2(Y,X))<=np.deg2rad(SECTOR_DEG))
        blue_map=matplotlib.colors.LinearSegmentedColormap.from_list('density',['#ffffff','#dbeafe','#93c5fd'])
        rgba=blue_map(np.clip(n/3,0,1))
        rgba[:,:,3]=mask.astype(float)
        im=ax.imshow(rgba,extent=[ORIGIN[0],ORIGIN[0]+SCALE*OUTER_R,
                                  ORIGIN[1]-.6*SCALE,ORIGIN[1]+.6*SCALE],
                     origin='lower',interpolation='bilinear',zorder=0)
        im.set_clip_path(clip)
        for outer,inner,color,alpha in ((FUEL_R,0,'#fef3c7',1),
                (SHELL_R,FUEL_R,'#e5e7eb',1),(SHELL_R,FUEL_R,ORANGE,.35)):
            p=Wedge(ORIGIN,SCALE*outer,-SECTOR_DEG,SECTOR_DEG,
                    width=SCALE*(outer-inner),fc=color,ec='none',alpha=alpha,zorder=1)
            p.set_clip_path(clip)
            ax.add_patch(p)
        for radius,color,style in ((CRITICAL_R,'#dc2626',':'),(CAPTURE_R,LINE,'--')):
            p=Arc(ORIGIN,2*SCALE*radius,2*SCALE*radius,theta1=-SECTOR_DEG,
                  theta2=SECTOR_DEG,color=color,lw=1.3,ls=style,zorder=3)
            p.set_clip_path(clip)
            ax.add_patch(p)
        for states,crossing in rays:
            points=xy(states[:,:2])
            ax.plot(points[:,0],points[:,1],color=BLUE,lw=1.8,zorder=4)
            arrow(tuple(points[100]),tuple(points[175]),color=BLUE,lw=1.8)
            source=xy(crossing[:2])
            ax.plot(*source,marker='*',ms=8,color=ORANGE,zorder=7)
        points=xy(continuation[:,:2])
        points[continuation[:,2]>0,1]+=SCALE*RETURN_OFFSET
        ax.plot(points[:,0],points[:,1],color=BLUE,lw=CONTINUATION_LW,ls='--',zorder=6)
        arrow(xy((RETURN_ARROW_RADII[0],RETURN_OFFSET)),
              xy((RETURN_ARROW_RADII[1],RETURN_OFFSET)),color=BLUE,
              lw=CONTINUATION_LW,mutation_scale=RETURN_ARROW_SIZE,zorder=7)
        for ray_index in (1,2,3):
            cross=rays[ray_index][1]
            source=cross[:2]
            axis_angle=np.arctan2(cross[3],cross[2])
            for angle in CHORD_ANGLES:
                a=axis_angle+np.deg2rad(angle)
                direction=np.array([np.cos(a),np.sin(a)])
                nearest=-np.dot(source,direction)
                impact_sq=max(0,np.dot(source,source)-nearest**2)
                length=nearest-np.sqrt(STOP_R**2-impact_sq) if impact_sq<=STOP_R**2 else nearest
                t=np.linspace(0,length,101)
                points=xy(source+t[:,None]*direction)
                segments=np.stack([points[:-1],points[1:]],axis=1)
                colors=np.tile(matplotlib.colors.to_rgba(ORANGE),(100,1))
                colors[:,3]=np.linspace(.95,.15,100)
                ax.add_collection(LineCollection(segments,colors=colors,linewidths=1,zorder=5))
        arrow(polar(CAPTURE_R,RADIAL_ANGLE),polar(RADIAL_STOP_R,RADIAL_ANGLE),
              color='#9a3412',lw=2.5)
        text(287,210,labels['centre'],ha='right')
        text(361,220,labels['fuel'],ha='center')
        text(451,207,labels['shell'],ha='center',rotation=90)
        text(730,355,labels['corona'],color='#64748b')
        text(*CRITICAL_LABEL_POS,labels['critical'],ha='right',color='#dc2626')
        ax.plot([CRITICAL_LEADER_START[0],polar(CRITICAL_R,SECTOR_DEG)[0]],
                [CRITICAL_LEADER_START[1],polar(CRITICAL_R,SECTOR_DEG)[1]],color='#dc2626',lw=.8)
        text(*CAPTURE_LABEL_POS,labels['capture'],fontsize=10,linespacing=1.4)
        ax.plot([CAPTURE_LEADER_START[0],polar(CAPTURE_R,SECTOR_DEG)[0]],
                [CAPTURE_LEADER_START[1],polar(CAPTURE_R,SECTOR_DEG)[1]],color=LINE,lw=.8)
        text(655,292,labels['power'],color=ORANGE)
        text(610,241,labels['continue'],color=BLUE,fontsize=10,linespacing=1.5)
        ax.plot([601,590,522],[241,231,215],color=BLUE,lw=.7,ls='--')
        text(150,246,labels['cone'],ha='center',color=ORANGE,linespacing=1.5)
        ax.plot([269,293,493],[246,260,249],color=ORANGE,lw=.8)
        text(154,154,labels['deposit'],ha='center',color='#9a3412',linespacing=1.5)
        ax.plot([289,320,451],[154,145,174],color='#9a3412',lw=.8)
        text(450,27,labels['radial'],ha='center',color='#9a3412',fontsize=10)
        ax.plot([281,281,polar(CAPTURE_R,RADIAL_ANGLE)[0]],
                [42,57,polar(CAPTURE_R,RADIAL_ANGLE)[1]],color='#9a3412',lw=.8)

        inset=fig.add_axes([46/W,325/H,242/W,154/H],facecolor='white',zorder=10)
        u=np.geomspace(.1,12,500)
        inset.set_xscale('log')
        inset.set_xlim(.1,12)
        inset.set_ylim(0,1)
        inset.axvspan(U_MIN,U_MAX,color='#ffedd5')
        inset.plot(u,np.exp(-u),color=ORANGE,lw=1.4)
        inset.vlines(edges,0,.075,color='#94a3b8',lw=.5)
        inset.set_xticks([.1,1,10],labels=['0.1','1','10'])
        inset.set_yticks([0,1])
        inset.tick_params(labelsize=8.5,pad=2)
        inset.set_xlabel(r'$E/T_h$',labelpad=0,fontsize=8.5)
        inset.set_ylabel(r'$dN/dE \propto e^{-E/T_h}$',labelpad=2,fontsize=8.5)
        inset.set_title(labels['spectrum'],fontsize=11,fontweight='semibold',pad=10)
        output=Path(f'docs/site/assets/hot-electron-generation-deposition-{lang}.png')
        fig.savefig(output,dpi=DPI,facecolor='white')
        with Image.open(output) as im:
            check(f'{lang} PNG size',im.size,im.size==(2*W,2*H))
            print(f'WROTE {output} {im.width}x{im.height}')
        plt.close(fig)


if __name__=='__main__':
    main()
