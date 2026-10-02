#!/usr/bin/env python3
"""Why genuine T4s of core failures are not TCs: emulate CrossCleanT4 (TrackCandidate.h:391-434: T4 removed if a
promoted T5/pT5 TC shares >=3 / >=2 of its 8 hits, or a pT3 TC's T3 shares >=2) by hit-coordinate matching.
T4 hits = its two T3s' 6 hits (t3_hit_*), T5 hits = its two T3s' hits; pT5 -> its T5; pT3 -> its T3."""
import collections, numpy as np, uproot
from load47 import ev, rows
from pair_attrib import bucket
F = '/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/Ntuple-files/LSTNtuple_s45_rebased_100evt.root'
t = uproot.open(F)['tree']
X = t.arrays([f't3_hit_{h}_{c}' for h in range(6) for c in 'xyz'] + ['t4_t3_idx0', 't4_t3_idx1', 'tc_simIdx'], library='np')
TY = {4: 'T5', 5: 'pT3', 7: 'pT5', 8: 'pLS', 9: 'T4'}


def t3hits(ie, i):
    return {tuple(np.round([X[f't3_hit_{h}_{c}'][ie][i] for c in 'xyz'], 3)) for h in range(6)}


c = collections.Counter()
for r in rows:
    if r['tc'] >= 0 or not r['t4']:
        continue
    ie = r['ev']; e = ev[ie]
    res = []
    for q in r['t4']:
        h4 = t3hits(ie, X['t4_t3_idx0'][ie][q]) | t3hits(ie, X['t4_t3_idx1'][ie][q])
        killer = None
        for j, ty in enumerate(e['tc_type']):
            ty = TY[int(ty)]
            if ty == 'T5':
                k5 = e['tc_t5Idx'][j]; th = t3hits(ie, e['t5_t3Idx0'][k5]) | t3hits(ie, e['t5_t3Idx1'][k5]); need = 3
            elif ty == 'pT5':
                k5 = e['pT5_t5Idx'][e['tc_pt5Idx'][j]]; th = t3hits(ie, e['t5_t3Idx0'][k5]) | t3hits(ie, e['t5_t3Idx1'][k5]); need = 2
            elif ty == 'pT3':
                th = t3hits(ie, e['pT3_t3Idx'][e['tc_pt3Idx'][j]]); need = 2
            else:
                continue
            n = len(h4 & th)
            if n >= need:
                killer = (ty, int(e['tc_isFake'][j]), int(X['tc_simIdx'][ie][j]) == r['sim'], n); break
        res.append(killer)
    if all(k is None for k in res):
        c['no CrossCleanT4 killer found -> T4 dedup (RemoveDupQuadruplets*) or TC cap'] += 1
    else:
        k = [x for x in res if x][0]
        c[f'CrossCleanT4 by {k[0]} TC ({"fake" if k[1] else ("same sim(<0.75? no)" if k[2] else "other sim")})'] += 1
    c['T4 displacedScore>0.5'] += any(e['t4_displacedScore'][q] > 0.5 for q in r['t4'])
    c['pT>100'] += r['pt'] > 100
for k, v in sorted(c.items()): print(f'{v:3d} {k}')
