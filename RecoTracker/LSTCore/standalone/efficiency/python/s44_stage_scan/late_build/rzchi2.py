"""Float32 vs float64 emulation of computePT3RZChiSquared (PixelTriplet.h:467-553) for genuine (pLS, T5-inner-T3) pairs.
rzChiSquared enters the pT3-DNN (log10) used with the pT5WP inside pT5 building (PixelQuintuplet.h:502)."""
import uproot, numpy as np, pickle, math, collections, warnings
warnings.filterwarnings('ignore')
from attrib3 import *
T=uproot.open('/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/trackingNtuple-100.root')['trackingNtuple/tree']
A=T.arrays(['sim_pt','see_stateTrajGlbX','see_stateTrajGlbY','see_stateTrajGlbZ'],library='np')
fp={np.asarray(A['sim_pt'][i],dtype=np.float32).tobytes():i for i in range(T.num_entries)}
k2=np.float64((2.99792458e-3*3.8)/2)
def rz(dt,x1,y1,z1,Px,Py,Pz,q,pts,types):
    f=dt; x1,y1,z1=f(x1/100),f(y1/100),f(z1/100); r1=f(math.hypot(x1,y1))
    Px,Py,Pz=f(Px),f(Py),f(Pz); a=f(-2*k2*100*q); p=np.sqrt(Px*Px+Py*Py+Pz*Pz); rou=a/p; R=f(0)
    for (x,y,z,endcap),mt in zip(pts,types):
        zsi=f(z/100); rtsi=f(math.hypot(x,y)/100)
        if endcap:
            s=(zsi-z1)*p/Pz
            X=x1+Px/a*np.sin(rou*s)-Py/a*(1-np.cos(rou*s)); Y=y1+Py/a*np.sin(rou*s)+Px/a*(1-np.cos(rou*s))
            res=abs(rtsi-np.sqrt(X*X+Y*Y))*100
        else:
            pA=r1*r1+2*(Px*Px+Py*Py)/(a*a)+2*(y1*Px-x1*Py)/a-rtsi*rtsi
            pB=2*(x1*Px+y1*Py)/a; pC=2*(y1*Px-x1*Py)/a+2*(Px*Px+Py*Py)/(a*a)
            AA=pB*pB+pC*pC; B=2*pA*pB; C=pA*pA-pC*pC; D=np.sqrt(B*B-4*AA*C)
            s1=(-B+D)/(2*AA); s2=(-B-D)/(2*AA)
            z1s=np.arcsin(s1)/rou*Pz/p+z1; z2s=np.arcsin(s2)/rou*Pz/p+z1
            res=np.nanmin([abs(z1s-zsi)*100,abs(z2s-zsi)*100]) if not(np.isnan(z1s) and np.isnan(z2s)) else np.nan
        e2=f(0.15**2) if mt==0 else f(5.0**2)
        R+=res*res/e2
    return float(np.sqrt(f(0.2)*R))
def pairs_of(r):
    e=ev[r['ev']]; e2=ev2[r['ev']]
    return [(p,q) for p in r['pls'] for q in r['t5'] if pairstage(e,e2,r['ev'],p,q)==7]
out=collections.defaultdict(list)
for r in rows:
    if not (r['dR']<0.02 and r['pls'] and r['t5']): continue
    P=pairs_of(r)
    if not P: continue
    grp=('built' if r['pt5'] else 'NOTbuilt')+('_pt>150' if r['pt']>150 else '_pt<150')
    e=ev[r['ev']]; ti=fp[np.asarray(e['sim_pt'],dtype=np.float32).tobytes()]
    vals=[]
    for p,q in P:
        s=plsinfo[r['ev']][p]['seed']; it3=e['t5_t3Idx0'][q]
        pts=[(e['t3_hit_%d_x'%k][it3],e['t3_hit_%d_y'%k][it3],e['t3_hit_%d_z'%k][it3],e['t3_hit_%d_layer'%k][it3]>6) for k in (0,2,4)]
        types=[e['t3_hit_%d_moduleType'%k][it3] for k in (0,2,4)]
        args=(A['see_stateTrajGlbX'][ti][s],A['see_stateTrajGlbY'][ti][s],A['see_stateTrajGlbZ'][ti][s],e['pLS_px'][p],e['pLS_py'][p],e['pLS_pz'][p],e['pLS_charge'][p],pts,types)
        vals.append((rz(np.float32,*args),rz(np.float64,*args),e['pLS_pt'][p]))
    out[grp].append((r,vals))
pickle.dump(dict(out),open('rzchi2.pkl','wb'))
for g in sorted(out):
    v=[x for r,vs in out[g] for x in vs]
    f32=np.array([x[0] for x in v]); f64=np.array([x[1] for x in v])
    rel=np.abs(f32-f64)/np.maximum(f64,1e-3)
    print(f'{g:18s} tracks={len(out[g]):4d} pairs={len(v):5d}  NaN32={np.isnan(f32).mean():.2f} NaN64={np.isnan(f64).mean():.2f} '
          f'median rz32={np.nanmedian(f32):.2f} rz64={np.nanmedian(f64):.2f}  frac |f32-f64|/f64>0.5: {(rel>0.5).mean():.2f}')
