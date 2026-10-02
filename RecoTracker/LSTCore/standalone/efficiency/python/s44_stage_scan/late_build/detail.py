import pickle,numpy as np,collections
from attrib3 import *
ex=pickle.load(open('attrib3_0.02.pkl','rb'))
def dump(k,n=40):
    print('\n##',k)
    print(' simPt  pLSpt  pLSptErr  T3ptEq  kind  quad  t5dnn  |eta|')
    for r,b in ex[k][:n]:
        st,p,q=b; e=ev[r['ev']]; it3=e['t5_t3Idx0'][q]
        print('%6.0f %6.1f %8.1f %7.1f  %s  %d  %.2f  %.2f'%(r['pt'],e['pLS_pt'][p],e['pLS_ptErr'][p],e['t3_radius'][it3]/kR1GeVf,t3kind(e,it3),e['pLS_isQuad'][p],e['t5_dnnScore'][q],abs(r['eta'])))
dump(ST[6])
# stage-6: pLS pt<5 (chi2 cuts apply)?
S6=ex[ST[7]]
lo=sum(1 for r,b in S6 if ev[r['ev']]['pLS_pt'][b[1]]<5); print('\nstage6 tracks N=%d, best pair pLS pt<5 (pT5 chi2 cuts active): %d'%(len(S6),lo))
print('stage6 pLS ptErr/pt median %.2f; T3ptEq/simPt median %.2f'%(np.median([ev[r['ev']]['pLS_ptErr'][b[1]]/ev[r['ev']]['pLS_pt'][b[1]] for r,b in S6]),
      np.median([ev[r['ev']]['t3_radius'][ev[r['ev']]['t5_t3Idx0'][b[2]]]/kR1GeVf/r['pt'] for r,b in S6])))
print('stage6 |eta| median %.2f, quad frac %.2f'%(np.median([abs(r['eta']) for r,b in S6]), np.mean([ev[r['ev']]['pLS_isQuad'][b[1]] for r,b in S6])))
# control: successes
ctl=[r for r in rows if r['dR']<0.02 and r['tc']>=0 and r['pls'] and r['pt5']]
print('control median simPt %.0f'%np.median([r['pt'] for r in ctl]))
for lo_,hi_ in [(0,20),(20,50),(50,100),(100,200),(200,5000)]:
    c=[r for r in rows if r['dR']<0.02 and lo_<=r['pt']<hi_ and r['pls']]
    ok=sum(1 for r in c if r['pt5']); rad=sum(1 for r in c if classify(r)[0]==ST[6])
    print(f' simPt [{lo_},{hi_}): core tracks w/ genuine pLS={len(c)}  genuine pT5 built={ok} ({100*ok/max(1,len(c)):.0f}%)  radius-stage-fail={rad}')
