"""Map each LST-ntuple pLS to its trackingNtuple seed (event matched by sim_pt fingerprint,
pLS matched by pixel-hit coordinates), and compute superbin + pixelType exactly as LSTPrepareInput.h."""
import uproot, numpy as np, pickle, math
ev=pickle.load(open('ev.pkl','rb'))
T=uproot.open('/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/trackingNtuple-100.root')['trackingNtuple/tree']
B=['sim_pt','pix_x','pix_y','pix_z','see_px','see_py','see_pz','see_dxy','see_dz','see_ptErr','see_q','see_algo','see_hitIdx','see_hitType',
   'see_stateTrajGlbPx','see_stateTrajGlbPy','see_stateTrajGlbPz','see_stateTrajGlbX','see_stateTrajGlbY','see_stateTrajGlbZ']
a=T.arrays(B,library='np')
fp={}
for i in range(T.num_entries): fp[np.asarray(a['sim_pt'][i],dtype=np.float32).tobytes()]=i
out=[]
nmiss=0; nfound=0
for ie,e in enumerate(ev):
    ti=fp.get(np.asarray(e['sim_pt'],dtype=np.float32).tobytes())
    assert ti is not None, ie
    px,py,pz=a['pix_x'][ti],a['pix_y'][ti],a['pix_z'][ti]
    key={}
    for s in range(len(a['see_px'][ti])):
        if a['see_algo'][ti][s] not in (4,22): continue
        h=a['see_hitIdx'][ti][s]; ht=a['see_hitType'][ti][s]
        if len(h)<3 or any(t!=0 for t in ht[:3]): continue
        k=tuple(np.round([px[h[0]],py[h[0]],pz[h[0]],px[h[1]],py[h[1]],pz[h[1]],px[h[2]],py[h[2]],pz[h[2]]],3))
        key.setdefault(k,[]).append(s)
    info=[]
    for p in range(len(e['pLS_pt'])):
        k=tuple(np.round([e['pLS_hit0_x'][p],e['pLS_hit0_y'][p],e['pLS_hit0_z'][p],e['pLS_hit1_x'][p],e['pLS_hit1_y'][p],e['pLS_hit1_z'][p],
                          e['pLS_hit2_x'][p],e['pLS_hit2_y'][p],e['pLS_hit2_z'][p]],3))
        cand=key.get(k,[])
        # disambiguate by ptIn
        best=None
        for s in cand:
            ptIn=math.hypot(a['see_stateTrajGlbPx'][ti][s],a['see_stateTrajGlbPy'][ti][s])
            if abs(ptIn-e['pLS_pt'][p])<1e-3*max(1,ptIn): best=s;break
        if best is None: nmiss+=1; info.append(None); continue
        nfound+=1; s=best
        P=np.array([a['see_px'][ti][s],a['see_py'][ti][s],a['see_pz'][ti][s]])
        pt=math.hypot(P[0],P[1]); eta=math.asinh(P[2]/pt); phi=math.atan2(P[1],P[0])
        dz=a['see_dz'][ti][s]
        etabin=int((eta+2.6)/((2*2.6)/25.)); phibin=int((phi+math.pi)/((2*math.pi)/72.)); dzbin=int((min(max(dz,-30),30)+30)/(2*30/25.))
        sb=int((25*72)*etabin+25*phibin+dzbin)
        ptIn=e['pLS_pt'][p]; ptErr=e['pLS_ptErr'][p]
        pr=np.array([a['see_stateTrajGlbPx'][ti][s],a['see_stateTrajGlbPy'][ti][s]]); r3=np.array([a['see_stateTrajGlbX'][ti][s],a['see_stateTrajGlbY'][ti][s]])
        dphi=math.remainder(math.atan2(r3[1],r3[0])-math.atan2(pr[1],pr[0]),2*math.pi)
        ptype=0 if ptIn>=2.0 else (1 if dphi>=0 else 2)
        info.append(dict(seed=int(s),sb=sb,etaPCA=eta,phiPCA=phi,dz=float(dz),ptype=ptype,etabin=etabin))
    e['_plsinfo']=info
print('pLS matched',nfound,'missed',nmiss)
pickle.dump([e['_plsinfo'] for e in ev],open('plsinfo.pkl','wb'))
