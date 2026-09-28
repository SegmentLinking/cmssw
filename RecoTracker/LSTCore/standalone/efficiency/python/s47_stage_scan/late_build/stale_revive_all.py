#!/usr/bin/env python3
"""Event-wide cost/benefit of clearing stale partOfPT5 on T5s (all T5s, not only core): T5s with partOfPT5=1, isDupBits==0,
all pT5s using them dedup-killed, and no surviving pT5 sharing >=4 of their hits (CrossCleanT5 would not kill them).
Classify: fake (t5_simIdx<0), genuine of a sim already matched by a TC (-> duplicate), genuine of an unmatched sim
(-> gain; core / all). Upper bound: RemoveDupQuintupletsBeforeTC among the revived T5s is not emulated."""
import collections, numpy as np, uproot
from load47 import ev, rows
F = '/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/Ntuple-files/LSTNtuple_s45_rebased_100evt.root'
X = uproot.open(F)['tree'].arrays(['t5_hitIndices', 'sim_tcIdx', 'sim_pt'], library='np')
core = {(r['ev'], r['sim']) for r in rows}
c = collections.Counter(); gain_sims = set(); gain_core = set()
for ie, e in enumerate(ev):
    alive = e['pT5_isDupReco'] == 0
    survHits = [set(np.asarray(X['t5_hitIndices'][ie][k]).tolist()) for k, a in zip(e['pT5_t5Idx'], alive) if a]
    users = collections.defaultdict(list)
    for i, k in enumerate(e['pT5_t5Idx']): users[int(k)].append(i)
    for q in np.nonzero(e['t5_partOfPT5'])[0]:
        if e['t5_isDupBits'][q] != 0 or any(alive[i] for i in users[q]):
            continue
        h = set(np.asarray(X['t5_hitIndices'][ie][q]).tolist())
        if any(len(h & sh) >= 4 for sh in survHits):
            c['stale, blocked by surviving pT5'] += 1; continue
        s = e['t5_simIdx'][q]
        if s < 0:
            c['REVIVABLE fake T5'] += 1
        elif X['sim_tcIdx'][ie][s] >= 0:
            c['REVIVABLE genuine, sim already matched (dup)'] += 1
        else:
            c['REVIVABLE genuine, sim UNMATCHED'] += 1; gain_sims.add((ie, s))
            if (ie, s) in core: gain_core.add((ie, s))
for k, v in sorted(c.items()): print(f'{v:5d} {k}')
print('distinct unmatched sims gained (all sims incl. non-denominator):', len(gain_sims), ' of which core-denominator:', len(gain_core))
