#!/usr/bin/env python3
"""Counterfactual: T5s whose ONLY pT5s were killed by pT5 dedup keep partOfPT5=1 forever
(PixelQuintuplet.h:747 set at build, never cleared) -> barred from AddT5asTrackCandidate
(TrackCandidate.h:634) and still exert isPT5-priority in BeforeTC; CrossCleanT5 also compares
against dead pT5s (no isDup check, TrackCandidate.h:253-281).
Estimate: release such T5s (clear partOfPT5 if all its pT5s dead, skip dead pT5s in CrossCleanT5),
require tightCutFlag, apply CrossCleanT5 vs ALIVE pT5 T5s (embedding cond) and BeforeTC-like cond vs
alive-pT5 T5s (priority), then score-dedup among released (greedy by t5_score). Count sims gained."""
import sys, math, collections, numpy as np, uproot
sys.path.insert(0, sys.argv[0].rsplit('/',1)[0])
from killed_pt5_attrib import hkey, dphi
FN=sys.argv[1]; NEV=int(sys.argv[2])
B=['pT5_t5Idx','pT5_isDupReco','t5_partOfPT5','t5_tightCutFlag','t5_isDupBits','t5_eta','t5_phi','t5_embed','t5_score',
   't5_simIdxAll','t5_simIdxAllFrac','t5_isFake','t5_t3Idx0','t5_t3Idx1','t3_lsIdx0','t3_lsIdx1','ls_mdIdx0','ls_mdIdx1',
   'md_anchor_x','md_anchor_y','md_anchor_z','md_other_x','md_other_y','md_other_z',
   'sim_q','sim_pt','sim_eta','sim_vx','sim_vy','sim_vz','sim_genjet_idx','sim_genjet_deltaR','genjet_pt','genjet_eta','sim_tcIdx']
t=uproot.open(FN)['tree']; C=collections.Counter()
for s0 in range(0,NEV,10):
  A=t.arrays(B,entry_start=s0,entry_stop=min(NEV,s0+10))
  for ev in A:
    g=lambda k: np.asarray(ev[k])
    p5=g('pT5_t5Idx').astype(int); pd=g('pT5_isDupReco').astype(bool)
    alive_t5=set(p5[~pd].tolist()); dead_t5=set(p5[pd].tolist())-alive_t5
    part=g('t5_partOfPT5').astype(bool); tight=g('t5_tightCutFlag').astype(bool); bits=g('t5_isDupBits')
    eta=g('t5_eta'); phi=g('t5_phi'); sc=g('t5_score'); fk=g('t5_isFake').astype(bool)
    emb=ev['t5_embed']
    ax,ay,az,ox,oy,oz=[g(k) for k in ('md_anchor_x','md_anchor_y','md_anchor_z','md_other_x','md_other_y','md_other_z')]
    l0,l1,a0,a1,q0,q1=[g(k) for k in ('ls_mdIdx0','ls_mdIdx1','t3_lsIdx0','t3_lsIdx1','t5_t3Idx0','t5_t3Idx1')]
    def hits(t5):
      ms=[]
      for t3 in (q0[t5],q1[t5]):
        for l in (a0[t3],a1[t3]):
          for m in (int(l0[l]),int(l1[l])):
            if m not in ms: ms.append(m)
      h=set()
      for m in ms: h|={hkey(ax[m],ay[m],az[m]),hkey(ox[m],oy[m],oz[m])}
      return h
    cand=[k for k in dead_t5 if tight[k] and not (bits[k]&1)]
    C['released_candidates']+=len(cand)
    E=lambda k: np.asarray(emb[k],dtype=float)
    al=list(alive_t5)
    surv=[]
    for k in cand:
      killed=False
      for j in al:
        de=abs(eta[k]-eta[j]); dp=abs(dphi(phi[k],phi[j]))
        if de>0.1 or dp>0.1: continue
        dr2=de*de+dp*dp; d2=float(((E(k)-E(j))**2).sum())
        if (dr2<0.02 and d2<0.1) or (dr2<1e-3 and d2<1.0) or (len(hits(k)&hits(j))>=5 and d2<1.0):
          killed=True; break
      if not killed: surv.append(k)
    # dedup among released: greedy by t5_score, BeforeTC-like cond
    surv.sort(key=lambda k: sc[k]); kept=[]
    for k in surv:
      ok=True
      for j in kept:
        de=abs(eta[k]-eta[j]); dp=abs(dphi(phi[k],phi[j]))
        if de>0.1 or dp>0.1: continue
        dr2=de*de+dp*dp; d2=float(((E(k)-E(j))**2).sum())
        if ((dr2<1e-3 or len(hits(k)&hits(j))>=5) and d2<1.0) or (dr2<0.02 and d2<0.1): ok=False; break
      if ok: kept.append(k)
    C['released_kept']+=len(kept); C['released_kept_fake']+=int(sum(fk[k] for k in kept))
    stc=g('sim_tcIdx'); dR=g('sim_genjet_deltaR'); gi=g('sim_genjet_idx'); gpt=g('genjet_pt'); ge=g('genjet_eta')
    simk=collections.Counter()
    for k in kept:
      for s,f in zip(ev['t5_simIdxAll'][k],ev['t5_simIdxAllFrac'][k]):
        if f>0.75: simk[int(s)]+=1
    for s,n in simk.items():
      if s>=len(stc): continue
      if stc[s]>=0: C['released_dup_of_existingTC']+=n; continue
      C['released_dup_extra']+=n-1
      ok=(ev['sim_q'][s]!=0 and ev['sim_pt'][s]>0.9 and abs(ev['sim_eta'][s])<4.5 and abs(ev['sim_vz'][s])<30 and math.hypot(ev['sim_vx'][s],ev['sim_vy'][s])<2.5 and gi[s]>=0 and gpt[gi[s]]>1000 and abs(ge[gi[s]])<2.5)
      if ok:
        C['gain_den']+=1
        if dR[s]<0.02: C['gain_core002']+=1
        if dR[s]<0.10: C['gain_core010']+=1
print(dict(C))
