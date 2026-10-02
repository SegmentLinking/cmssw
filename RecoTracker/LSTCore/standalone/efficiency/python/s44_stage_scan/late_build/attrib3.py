"""Kernel-order waterfall per (genuine pLS x genuine T5) pair; track stage = furthest stage any pair reaches.
Kernel order (CreatePixelQuintupletsFromMap): pixelType/superbin (LSTEvent.dev.cc:1118) -> pixel map & 2S veto
(PixelQuintuplet.h:662-666) -> pLS isDup first pass (668; Kernels.h:503) -> T5 AfterBuild isDup (682)
-> passRadiusCriterion (PixelTriplet.h:584) -> [not emulated: PPBB/PPEE tracklet x2 (594-614), pT3-DNN pT5WP (682),
pT5 rz/rphi chi2 only if pLS pT<5 (PixelQuintuplet.h:576,609), rphi-inwards if T5 regR<5GeV (622)]."""
import pickle, numpy as np, collections, sys
from attrib2 import ev, ev2, rows, plsinfo, connected, radius_pass, t3kind, kR1GeVf
import os
_V=os.environ.get('PLSDUP','cur')
dup=pickle.load(open('plsdup.pkl','rb')) if _V in('cur','none') else pickle.load(open('plsdup_variants.pkl','rb'))[_V]
if _V=='none': dup=[[False]*len(x) for x in pickle.load(open('plsdup.pkl','rb'))]
ST=['0 no genuine T5 (genuine T3 exists)','0 no genuine T5, no genuine T3',
    '1 pLS invalid superbin/type','2 map: T5 module not connected / 2S','3 ALL genuine pLS flagged isDup (CheckHitspLS 1st pass)',
    '4 genuine T5 AfterBuild-isDup','5 radius criterion fails','6 passes emulated cuts (tracklet/DNN/chi2 not emulated)']
def pairstage(e,e2,ie,p,q):
    pi=plsinfo[ie][p]
    if not (0<=pi['sb']<45000): return 2
    it3=e['t5_t3Idx0'][q]
    if e['t3_hit_0_moduleType'][it3]!=0 or not connected(pi,[int(e2['t3_hit_0_detId'][it3]),int(e2['t3_hit_1_detId'][it3])]): return 3
    if dup[ie][p]: return 4
    if e['t5_isDupBits'][q]&1: return 5
    if not radius_pass(e['pLS_pt'][p]*kR1GeVf,e['pLS_ptErr'][p]*kR1GeVf,e['t3_radius'][it3],t3kind(e,it3)): return 6
    return 7
def classify(r):
    e=ev[r['ev']]; e2=ev2[r['ev']]
    if not r['t5']: return (ST[0] if r['t3'] else ST[1]), None
    best=max((pairstage(e,e2,r['ev'],p,q),p,q) for p in r['pls'] for q in r['t5'])
    return ST[best[0]], best
if __name__=='__main__':
    CUT=float(sys.argv[1]) if len(sys.argv)>1 else 0.02
    groups=[('FAIL core, genuine pLS & no genuine pT5',lambda r:r['tc']<0 and r['pls'] and not r['pt5']),
            ('SUCCESS core via genuine pT5 (control)',lambda r:r['tc']>=0 and r['pls'] and r['pt5']),
            ('FAIL OUTSIDE core (0.1<dR<0.4), genuine pLS & no pT5',None)]
    for lab,sel in groups:
        if sel is None: S=[r for r in rows if 0.1<r['dR']<0.4 and r['tc']<0 and r['pls'] and not r['pt5']]
        else: S=[r for r in rows if r['dR']<CUT and sel(r)]
        c=collections.Counter(); ex=collections.defaultdict(list)
        for r in S:
            k,b=classify(r); c[k]+=1; ex[k].append((r,b))
        print(f'\n=== {lab} (dR<{CUT} unless stated): N={len(S)}')
        for k in ST:
            if c[k]: print(f'  {c[k]:4d} ({100*c[k]/max(1,len(S)):5.1f}%)  {k}  | median simPt {np.median([x[0]["pt"] for x in ex[k]]):.0f}')
        if lab.startswith('FAIL core'): pickle.dump(ex,open('attrib3_%s.pkl'%CUT,'wb'))
