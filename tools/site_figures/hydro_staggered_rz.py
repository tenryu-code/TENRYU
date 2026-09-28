"""Staggered RZ cell, bilinear mass lumps, velocities and corner forces.

Display 720x480, output 1440x960, dpi 200. CCW nodes are (1,0),
(1.95,.18), (2.10,1.05), (.92,.88); centre is their mean. Reference
square [-1,1]^2 uses bilinear N_k, x=sum N_k x_k, J=det(dx/d(xi,eta)).
Mass fractions are integral(N_k R J)/integral(R J), with 4x4 and 2x2
Gauss quadrature. V=(pi/3) sum (R_k+R_next)(R_k Z_next-R_next Z_k).
S_k=dV/dx_k; central-difference h=1e-6, max component relative error<1e-7.
u_k=.30(x_k-x_c)+(.05,.02); positive common pressure factor is absorbed
in the force drawing scale, max arrow length=.45. Sum tolerance=1e-12;
axial rigid translation u=(0,1), tolerance=1e-12; outward dot products>0.
"""
from pathlib import Path
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
from PIL import Image

W,H,DPI=720,480,200
NODES=np.array([[1.,0.],[1.95,.18],[2.10,1.05],[.92,.88]])
H_DIFF, FD_TOL, SUM_TOL = 1e-6,1e-7,1e-12
VELOCITY_RATE, VELOCITY_OFFSET, MAX_FORCE_LENGTH = .30,np.array([.05,.02]),.45
OUT=Path('docs/site/assets/hydro-staggered-rz.png')
INK,SLATE,BLUE,ORANGE,SECONDARY='#1e293b','#475569','#2563eb','#ea580c','#64748b'
COLORS=['#dbeafe','#dcfce7','#fef3c7','#ede9fe']
warnings.filterwarnings('error',message=r'Glyph .* missing')
plt.rcParams.update({'font.family':['Hiragino Sans','DejaVu Sans'],'font.size':10,
 'mathtext.fontset':'stix','axes.linewidth':.9,'axes.edgecolor':SLATE,
 'axes.labelcolor':INK,'text.color':INK,'xtick.color':SLATE,'ytick.color':SLATE,
 'figure.facecolor':'white','savefig.facecolor':'white'})


def volume(x):
    y=np.roll(x,-1,axis=0)
    return np.pi/3*np.sum((x[:,0]+y[:,0])*(x[:,0]*y[:,1]-y[:,0]*x[:,1]))


def gradient(x):
    s=np.zeros_like(x)
    for k in range(4):
        j=(k+1)%4
        r,z=x[k]; t,w=x[j]
        cross=r*w-t*z
        s[k]+=np.pi/3*np.array([cross+(r+t)*w,-(r+t)*t])
        s[j]+=np.pi/3*np.array([cross-(r+t)*z,(r+t)*r])
    return s


def lumps(order):
    points,weights=np.polynomial.legendre.leggauss(order)
    signs=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
    integrals=np.zeros(4)
    for xi,wi in zip(points,weights):
        for eta,wj in zip(points,weights):
            a,b=signs.T
            basis=(1+a*xi)*(1+b*eta)/4
            derivatives=np.column_stack((a*(1+b*eta),b*(1+a*xi)))/4
            jac=np.linalg.det(NODES.T@derivatives)
            radius=basis@NODES[:,0]
            integrals+=wi*wj*basis*radius*jac
    return integrals/integrals.sum()


def main():
    center=NODES.mean(axis=0)
    s=gradient(NODES); v=volume(NODES)
    u=VELOCITY_RATE*(NODES-center)+VELOCITY_OFFSET
    f4,f2=lumps(4),lumps(2)
    fd=np.zeros_like(s)
    for k in range(4):
        for d in range(2):
            delta=np.zeros_like(NODES); delta[k,d]=H_DIFF
            fd[k,d]=(volume(NODES+delta)-volume(NODES-delta))/(2*H_DIFF)
    error=np.max(np.abs(fd-s)/np.abs(s))
    outward=np.einsum('ij,ij->i',s,NODES-center)
    axial=s[:,1].sum()
    checks=[('4x4 mass fractions',f4,abs(f4.sum()-1)<SUM_TOL),
            ('2x2 mass fractions',f2,abs(f2.sum()-1)<SUM_TOL),
            ('sum fractions - 1, tolerance 1e-12',f4.sum()-1,abs(f4.sum()-1)<SUM_TOL),
            ('max 4x4 vs 2x2 difference, tolerance 1e-12',abs(f4-f2).max(),abs(f4-f2).max()<SUM_TOL),
            ('S finite-difference max relative error < 1e-7',error,error<FD_TOL),
            ('outward S dot (node-center) > 0',outward,np.all(outward>0)),
            ('sum S dot (0,1), tolerance 1e-12',axial,abs(axial)<SUM_TOL),
            ('revolved volume > 0',v,v>0)]
    for name,value,passed in checks:
        rendered=np.array2string(value,precision=14) if isinstance(value,np.ndarray) else f'{value:.15g}'
        print(f'CHECK {name}: {rendered} {"PASS" if passed else "FAIL"}')
    if not all(c[2] for c in checks):
        raise SystemExit(1)
    fig=plt.figure(figsize=(W/100,H/100),dpi=DPI)
    ax=fig.add_axes([12/W,80/H,432/W,329/H])
    ax.set(xlim=(-.12,2.9),ylim=(-.65,1.65),aspect='equal'); ax.axis('off')
    ax.plot([0,0],[-.5,1.55],ls='-.',color='#94a3b8',lw=1)
    ax.text(.06,1.55,r'axis $R=0$',fontsize=10,va='center')
    origin=np.array([.16,-.43])
    for end,label,offset in [(origin+[.4,0],r'$R$',(.03,-.025)),(origin+[0,.4],r'$Z$',(-.035,.04))]:
        ax.add_patch(FancyArrowPatch(origin,end,arrowstyle='-|>',mutation_scale=12,lw=1.3,color=SLATE))
        ax.text(*(end+offset),label,fontsize=10)
    mid=(NODES+np.roll(NODES,-1,axis=0))/2
    for k in range(4):
        poly=np.array([NODES[k],mid[k],center,mid[(k-1)%4]])
        ax.add_patch(Polygon(poly,closed=True,facecolor=COLORS[k],edgecolor='#cbd5e1',lw=1))
        point=poly.mean(axis=0)
        ax.text(*point,rf'$m_{{c,{k}}}$ ='+'\n'+rf'${f4[k]:.4f}\,M_c$',ha='center',va='center',fontsize=10,linespacing=1.15)
    ax.add_patch(Polygon(NODES,closed=True,fill=False,edgecolor=SLATE,lw=1.3))
    ax.plot(*center,'o',color='black',ms=4,zorder=6)
    ax.annotate(r'$\rho, e, P, T$'+'\n(cell-centered)',xy=center,xytext=(2.24,.70),
        fontsize=10,va='center',arrowprops={'arrowstyle':'-','color':SLATE,'lw':.8})
    force=s*MAX_FORCE_LENGTH/np.linalg.norm(s,axis=1).max()
    node_offsets=[(.08,-.12),(.07,.05),(.06,-.08),(-.17,-.07)]
    vel_offsets=[(-.22,.05),(.04,-.11),(.04,.05),(-.25,-.06)]
    force_offsets=[(-.13,-.13),(.02,-.13),(.01,.03),(-.24,.04)]
    for k in range(4):
        x=NODES[k]
        ax.plot(*x,'o',color=INK,ms=4,zorder=7)
        ax.text(*(x+node_offsets[k]),rf'$n_{k}$',fontsize=10)
        for vector,color in [(force[k],ORANGE),(u[k],BLUE)]:
            ax.add_patch(FancyArrowPatch(x,x+vector,arrowstyle='-|>',mutation_scale=12,
                                        shrinkA=0,shrinkB=0,lw=1.3,color=color,zorder=5))
        ax.text(*(x+u[k]+vel_offsets[k]),rf'$\mathbf{{u}}_{{n_{k}}}$',fontsize=10,color=BLUE)
        ax.text(*(x+force[k]+force_offsets[k]),rf'$\mathbf{{F}}_{{c\to n_{k}}}$',fontsize=10,color=ORANGE)
    text_block='\n'.join([
        'corner force',
        r'$\mathbf{F}_{c\to n_k} = +(P_c+Q_c)\,\mathbf{S}_{c,k}$',
        r'$\mathbf{S}_{c,k} = \partial V_c/\partial\mathbf{x}_{c,k}$',
        '(exact revolved volume)',
        'points outward',
        r'$\dot V_c = \sum_k \mathbf{u}_{n_k}\cdot\mathbf{S}_{c,k}$'])
    cell_center_y=fig.transFigure.inverted().transform(ax.transData.transform(center))[1]
    fig.text(480/W,cell_center_y,text_block,fontsize=9,linespacing=1.65,va='center')
    fig.text(35/W,75/H,'node velocity (positions and velocities live at nodes)',fontsize=8.5,color=BLUE)
    fig.savefig(OUT,dpi=DPI,facecolor='white'); plt.close(fig)
    with Image.open(OUT) as im:
        print(f'WROTE {OUT} {im.width}x{im.height}')
        if im.size != (2*W,2*H):
            raise SystemExit('FAIL pixel dimensions')

if __name__=='__main__':
    main()
