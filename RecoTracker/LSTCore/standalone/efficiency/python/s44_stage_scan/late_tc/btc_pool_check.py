#!/usr/bin/env python3
"""For pool tracks (core, genuine pT5 built, no TC): genuine standalone T5s killed by BeforeTC.
If BeforeTC spared them (e.g. `isDup & 1` -> `& 3` chain fix), would they survive CrossCleanT5 vs alive pT5s
and tightCutFlag?  Upper bound on defect (i) for this bucket."""
import sys, json, collections, numpy as np, uproot
sys.path.insert(0, sys.argv[0].rsplit('/',1)[0])
import numpy as _np
def dphi(a, b): return (a - b + _np.pi) % (2 * _np.pi) - _np.pi
FN, RJ = sys.argv[1], sys.argv[2]
rows = json.load(open(RJ)); byev = collections.defaultdict(list)
for r in rows: byev[r['ev']].append(r['sim'])
B=['pT5_t5Idx','pT5_isDupReco','t5_partOfPT5','t5_tightCutFlag','t5_isDupBits','t5_eta','t5_phi','t5_embed','sim_t5IdxAll','sim_t5IdxAllFrac']
t=uproot.open(FN)['tree']; C=collections.Counter()
for e, sims in sorted(byev.items()):
  ev=t.arrays(B,entry_start=e,entry_stop=e+1)[0]; g=lambda k: np.asarray(ev[k])
  p5=g('pT5_t5Idx').astype(int); pd=g('pT5_isDupReco').astype(bool); al=set(p5[~pd].tolist())
  bits=g('t5_isDupBits'); part=g('t5_partOfPT5'); tight=g('t5_tightCutFlag'); eta=g('t5_eta'); phi=g('t5_phi'); emb=ev['t5_embed']
  for s in sims:
    gt=[int(k) for k,f in zip(ev['sim_t5IdxAll'][s],ev['sim_t5IdxAllFrac'][s]) if f>0.75]
    btc=[k for k in gt if not part[k] and not (bits[k]&1) and (bits[k]&0x0E)]
    if not btc: continue
    C['tracks_with_BTC_killed_genuineT5(>0.75)']+=1
    ok=False
    for k in btc:
      if not tight[k]: continue
      kill=False
      for j in al:
        de=abs(eta[k]-eta[j]); dp=abs(dphi(phi[k],phi[j])); dr2=de*de+dp*dp
        d2=float(((np.asarray(emb[k])-np.asarray(emb[j]))**2).sum())
        if (dr2<0.02 and d2<0.1) or (dr2<1e-3 and d2<1.0): kill=True; break
      if not kill: ok=True
    C['...would_survive_tight+CrossCleanT5']+=int(ok)
print(dict(C))
