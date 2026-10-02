"""Exact offline emulation of CheckHitspLS first pass (Kernels.h:503-583, secondpass=false), which runs
BEFORE createPixelQuintuplets/Triplets (LST.cc:96) and whose isDup flag makes pT5/pT3 builders skip the pLS
(PixelQuintuplet.h:668, PixelTriplet.h:716).  pLSHitsIdxs = {h0,h2,h1,h3|h2} (Segment.h:1098); score from Segment.h:1091-1096."""
import uproot, numpy as np, pickle, math
ev=pickle.load(open('ev.pkl','rb')); plsinfo=pickle.load(open('plsinfo.pkl','rb'))
T=uproot.open('/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/trackingNtuple-100.root')['trackingNtuple/tree']
a=T.arrays(['sim_pt','see_px','see_py','see_pz','see_dxy','see_dz','see_hitIdx','see_stateTrajGlbX','see_stateTrajGlbY','see_stateTrajGlbZ'],library='np')
fp={np.asarray(a['sim_pt'][i],dtype=np.float32).tobytes():i for i in range(T.num_entries)}
def r3pca(px,py,pz,dxy,dz):
    pt=math.hypot(px,py); p=math.sqrt(pt*pt+pz*pz)
    return (-dxy*py/pt-px/p*pz/p*dz, dxy*px/pt-py/p*pz/p*dz, dz*pt*pt/p/p)
dup=[]; ntot=0; nflag=0
for ie,e in enumerate(ev):
    ti=fp[np.asarray(e['sim_pt'],dtype=np.float32).tobytes()]
    n=len(e['pLS_pt']); H=[]; S=np.zeros(n,np.float32); Q=np.array(e['pLS_isQuad'],bool); ETA=np.array(e['pLS_eta'],np.float32)
    for p in range(n):
        s=plsinfo[ie][p]['seed']; h=list(a['see_hitIdx'][ti][s])
        H.append((h[0],h[2],h[1],h[3] if len(h)>3 else h[2]))
        px,py,pz=a['see_px'][ti][s],a['see_py'][ti][s],a['see_pz'][ti][s]
        x0,y0,z0=r3pca(px,py,pz,a['see_dxy'][ti][s],a['see_dz'][ti][s])
        eta=math.asinh(pz/math.hypot(px,py)); slope=math.sinh(eta)
        icpt=z0-slope*math.hypot(x0,y0)
        xl,yl,zl=a['see_stateTrajGlbX'][ti][s],a['see_stateTrajGlbY'][ti][s],a['see_stateTrajGlbZ'][ti][s]
        S[p]=np.float32((math.hypot(xl,yl)*slope+icpt-zl)**2)
    flag=np.zeros(n,bool)
    for ix in range(n):
        s1=H[ix]
        for jx in range(ix+1,n):
            if abs(ETA[jx]-ETA[ix])>0.1: continue
            s2=set(H[jx]); m=sum(1 for h in s1 if h in s2)
            if m>=3:
                qd=int(Q[ix])-int(Q[jx]); sd=S[ix]-S[jx]
                rm = jx if qd>0 else ix if qd<0 else (jx if sd<0 else ix)
                flag[rm]=True
    dup.append(flag); ntot+=n; nflag+=flag.sum()
print('pLS total',ntot,'flagged first-pass isDup',nflag, '%.1f%%'%(100*nflag/ntot))
pickle.dump(dup,open('plsdup.pkl','wb'))
