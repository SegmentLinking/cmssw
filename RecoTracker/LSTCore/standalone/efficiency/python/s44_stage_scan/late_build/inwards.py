"""Would a T5-circle -> pixel-hit residual ("inwards", cf. rPhiChiSquaredInwards PixelQuintuplet.h:620, only cut when T5 R<5GeV)
pick the genuine T5 over the foreign T5 that stole the genuine pLS?  Circle = algebraic LSQ fit to the 5 T5 anchor hits
(approximates regressionCenter/Radius). Residual = RMS over pLS pixel hits of |dist-R| in microns."""
import numpy as np, collections, uproot
from attrib3 import ev, rows
def fit(x,y):
    A=np.c_[x,y,np.ones_like(x)]; b=-(x*x+y*y); s=np.linalg.lstsq(A,b,rcond=None)[0]
    g,f=-s[0]/2,-s[1]/2; return g,f,np.sqrt(g*g+f*f-s[2])
def t5pts(e,q):
    i0,i1=e['t5_t3Idx0'][q],e['t5_t3Idx1'][q]
    X=[e['t3_hit_%d_x'%k][i0] for k in (0,2,4)]+[e['t3_hit_%d_x'%k][i1] for k in (2,4)]
    Y=[e['t3_hit_%d_y'%k][i0] for k in (0,2,4)]+[e['t3_hit_%d_y'%k][i1] for k in (2,4)]
    return np.array(X,float),np.array(Y,float)
def resid(e,p,q):
    g,f,R=fit(*t5pts(e,q))
    px=np.array([e["pLS_hit%d_x"%k][p] for k in range(4)]); py=np.array([e["pLS_hit%d_y"%k][p] for k in range(4)]); ok=np.hypot(px,py)<30; px,py=px[ok],py[ok]
    return 1e4*np.sqrt(np.mean((np.hypot(px-g,py-f)-R)**2))
B=[r for r in rows if r['dR']<0.02 and r['tc']<0 and r['pls'] and not r['pt5'] and r['t5']]
win=collections.Counter(); G=[];Fo=[]
for r in B:
    e=ev[r['ev']]; pls=set(r['pls'])
    ids=[i for i,p in enumerate(e['pT5_plsIdx']) if p in pls]
    if not ids: continue
    for i in ids:
        p=e['pT5_plsIdx'][i]; qf=e['pT5_t5Idx'][i]
        if qf in r['t5']: continue
        rf=resid(e,p,qf); rg=min(resid(e,p,q) for q in r['t5'])
        G.append(rg); Fo.append(rf); win['genuine T5 closer' if rg<rf else 'foreign T5 closer']+=1
print('pairs (stolen pT5 vs best genuine T5 with same pLS):',dict(win))
print('median residual um: genuine %.0f  foreign %.0f'%(np.median(G),np.median(Fo)))
# reference: built genuine pT5 pairs residual distribution (to set a cut)
ref=[]
for r in rows:
    if r['dR']<0.02 and r['pt5'] and r['pt']>50:
        e=ev[r['ev']]
        for i in r['pt5'][:2]: ref.append(resid(e,e['pT5_plsIdx'][i],e['pT5_t5Idx'][i]))
print('reference genuine built pT5 (core, pt>50) residual um p50/p90/p99: %.0f/%.0f/%.0f'%tuple(np.percentile(ref,[50,90,99])))
