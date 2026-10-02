"""T5-circle->pixel residual (um) as replacement high-pT pairing gate: genuine pairs in radius/DNN-failed tracks vs
built genuine pT5 (reference) vs foreign 'stolen' pairings. Line fit used when circle fit is ill-conditioned (R>1e5 cm)."""
import numpy as np, pickle, collections
import attrib3 as A, pt3dnn as D
from inwards import t5pts
def resid(e,p,q):
    x,y=t5pts(e,q); px=np.array([e["pLS_hit%d_x"%k][p] for k in range(4)]); py=np.array([e["pLS_hit%d_y"%k][p] for k in range(4)]); ok=np.hypot(px,py)<30; px,py=px[ok],py[ok]
    M=np.c_[x,y,np.ones_like(x)]; s=np.linalg.lstsq(M,-(x*x+y*y),rcond=None)[0]; g,f=-s[0]/2,-s[1]/2; R2=g*g+f*f-s[2]
    if R2>0 and np.sqrt(R2)<1e5: return 1e4*np.sqrt(np.mean((np.hypot(px-g,py-f)-np.sqrt(R2))**2))
    # straight line (total least squares)
    P=np.c_[x,y]; m=P.mean(0); u=np.linalg.svd(P-m)[2][0]; n=np.array([-u[1],u[0]])
    return 1e4*np.sqrt(np.mean(((np.c_[px,py]-m)@n)**2))
ex=pickle.load(open('attrib3_0.02.pkl','rb'))
gen=[]
for k in (A.ST[6],A.ST[7]):
    for r,b in ex[k]:
        e=A.ev[r['ev']]; ps=[p for p in r['pls'] if not A.dup[r['ev']][p]]
        gen.append(min(resid(e,p,q) for p in ps for q in r['t5'] if (e['t5_isDupBits'][q]&1)==0))
ref=[];frn=[]
for r in A.rows:
    if r['dR']<0.02 and r['pt']>50 and r['pt5']:
        e=A.ev[r['ev']]; i=r['pt5'][0]; ref.append(resid(e,e['pT5_plsIdx'][i],e['pT5_t5Idx'][i]))
for r in A.rows:
    if not(r['dR']<0.02 and r['tc']<0 and r['pls'] and not r['pt5']): continue
    e=A.ev[r['ev']]
    for i,p in enumerate(e['pT5_plsIdx']):
        if p in r['pls'] and e['pT5_t5Idx'][i] not in r['t5']: frn.append(resid(e,p,e['pT5_t5Idx'][i]))
pc=lambda v:np.percentile(v,[50,90,99]).round(0)
print('built genuine pT5 core pt>50 (N=%d) p50/p90/p99 um:'%len(ref),pc(ref))
print('radius+DNN/tracklet-failed tracks best genuine pair (N=%d):'%len(gen),pc(gen),' frac<300um=%.2f <1000um=%.2f'%(np.mean(np.array(gen)<300),np.mean(np.array(gen)<1000)))
print('foreign stolen pairings (N=%d):'%len(frn),pc(frn),' frac<300um=%.2f <1000um=%.2f'%(np.mean(np.array(frn)<300),np.mean(np.array(frn)<1000)))
