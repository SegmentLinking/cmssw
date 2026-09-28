#!/usr/bin/env python3
"""Reconcile the 'unstale' estimate with late_build (+14-18): for core failing sims whose genuine T5 is stale
(partOfPT5 set, all its pT5s dead), what happens after clearing the flag?  Uses the exact BTC/CC relations of q6."""
import pickle, collections
import numpy as np
from q6_combined import PK
E = pickle.load(open(PK, 'rb'))
C = collections.Counter()
for e in E:
    alive = set(np.nonzero(~e['dup'])[0].tolist())
    has_alive = np.zeros(e['n5'], bool)
    for k in alive: has_alive[e['p_t5'][k]] = True
    partn = e['part'] & has_alive
    stale = set(np.nonzero(e['part'] & ~has_alive)[0].tolist())
    for s, core, base in e['sims']:
        if not core or base: continue
        g = [t for t in range(e['n5']) if s in e['t5sims'][t]]
        gs = [t for t in g if t in stale]
        if not gs: continue
        C['failing core sims with a stale genuine T5'] += 1
        fates = []
        for t in gs:
            if any(not (partn[t] and partn[j]) for j in e['losers'].get(t, [])): fates.append('BTC')
            elif t in e['cc_pt3']: fates.append('CC_pT3')
            elif any(k in alive for k in e['cc_pt5'].get(t, [])):
                ks = [k for k in e['cc_pt5'][t] if k in alive]
                fates.append('CC_pT5_sameT5' if any(e['p_t5'][k] == t for k in ks) else 'CC_pT5_otherT5')
            else: fates.append('SURVIVES')
        for f in ['SURVIVES', 'CC_pT5_otherT5', 'CC_pT5_sameT5', 'CC_pT3', 'BTC']:
            if f in fates: C['  best fate after unstale: ' + f] += 1; break
for k, v in C.items(): print(f'{v:>5}  {k}')
