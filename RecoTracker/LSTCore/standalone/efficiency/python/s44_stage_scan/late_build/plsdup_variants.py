"""Why are ALL genuine pLS of 61 core tracks flagged? Compare current CheckHitspLS pass-1 with variants:
 V1 unique-hit counting (no triplet double count), V2 greedy (only an un-flagged winner can kill), V3 = V1+V2.
Also count pLS surviving overall (seed-supply proxy for pT5 fake/dup risk)."""
import uproot, numpy as np, pickle, math, collections
ev=pickle.load(open('ev.pkl','rb')); plsinfo=pickle.load(open('plsinfo.pkl','rb')); rows=pickle.load(open('rows.pkl','rb'))
cur=pickle.load(open('plsdup.pkl','rb'))
T=uproot.open('/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/trackingNtuple-100.root')['trackingNtuple/tree']
a=T.arrays(['sim_pt','see_px','see_py','see_pz','see_dxy','see_dz','see_hitIdx','see_stateTrajGlbX','see_stateTrajGlbY','see_stateTrajGlbZ'],library='np')
fp={np.asarray(a['sim_pt'][i],dtype=np.float32).tobytes():i for i in range(T.num_entries)}
def r3pca(px,py,pz,dxy,dz):
    pt=math.hypot(px,py); p=math.sqrt(pt*pt+pz*pz)
    return (-dxy*py/pt-px/p*pz/p*dz, dxy*px/pt-py/p*pz/p*dz, dz*pt*pt/p/p)
res={k:[] for k in ('cur','V1','V2','V3')}; why=[]
for ie,e in enumerate(ev):
    ti=fp[np.asarray(e['sim_pt'],dtype=np.float32).tobytes()]
    n=len(e['pLS_pt']); H=[]; S=np.zeros(n,np.float32); Q=np.array(e['pLS_isQuad'],bool); ETA=np.array(e['pLS_eta'],np.float32)
    for p in range(n):
        s=plsinfo[ie][p]['seed']; h=list(a['see_hitIdx'][ti][s]); H.append((h[0],h[2],h[1],h[3] if len(h)>3 else h[2]))
        px,py,pz=a['see_px'][ti][s],a['see_py'][ti][s],a['see_pz'][ti][s]
        x0,y0,z0=r3pca(px,py,pz,a['see_dxy'][ti][s],a['see_dz'][ti][s]); sl=math.sinh(math.asinh(pz/math.hypot(px,py)))
        xl,yl,zl=a['see_stateTrajGlbX'][ti][s],a['see_stateTrajGlbY'][ti][s],a['see_stateTrajGlbZ'][ti][s]
        S[p]=np.float32((math.hypot(xl,yl)*sl+z0-sl*math.hypot(x0,y0)-zl)**2)
    def pairs(uniq):
        P=[]
        for ix in range(n):
            s1=set(H[ix]) if uniq else H[ix]
            for jx in range(ix+1,n):
                if abs(ETA[jx]-ETA[ix])>0.1: continue
                s2=set(H[jx]); m=sum(1 for h in s1 if h in s2)
                if m>=3:
                    qd=int(Q[ix])-int(Q[jx]); sd=S[ix]-S[jx]
                    rm = jx if qd>0 else ix if qd<0 else (jx if sd<0 else ix)
                    P.append((ix+jx-rm, rm))  # (winner, loser)
        return P
    for tag,uniq,greedy in (('V1',True,False),('V2',False,True),('V3',True,True)):
        P=pairs(uniq)
        if not greedy:
            f=np.zeros(n,bool)
            for w,l in P: f[l]=True
        else:
            adj=collections.defaultdict(list)
            for w,l in P: adj[w].append(l); adj[l].append(w)
            order=sorted(range(n),key=lambda i:(-int(Q[i]),S[i],i))
            f=np.zeros(n,bool); kept=set()
            for i in order:
                if any(j in kept for j in adj[i]): f[i]=True
                else: kept.add(i)
        res[tag].append(f)
    res['cur'].append(cur[ie])
pickle.dump(res,open('plsdup_variants.pkl','wb'))
for k in res: 
    tot=sum(len(x) for x in res[k]); fl=sum(x.sum() for x in res[k]); print(f'{k}: flagged {fl}/{tot} = {100*fl/tot:.1f}%')
# per track: which core tracks (genuine pLS) keep >=1 unflagged genuine pLS
for CUT in (0.02,):
    core=[r for r in rows if r['dR']<CUT and r['pls']]
    for k in res:
        alive=sum(any(not res[k][r['ev']][p] for p in r['pls']) for r in core)
        fcore=[r for r in core if r['tc']<0 and not r['pt5']]
        falive=sum(any(not res[k][r['ev']][p] for p in r['pls']) for r in fcore)
        print(f'  dR<{CUT} {k}: core tracks w/ genuine pLS={len(core)}, keep an unflagged genuine pLS={alive}; in fail-bucket({len(fcore)}): {falive}')
