#!/usr/bin/env python3
"""Stale partOfPT5 (set at pT5 creation, PixelQuintuplet.h:777-780; never cleared when RemoveDupPixelQuintupletsFromMap,
Kernels.h:735-770, kills the pT5). Effects: T5 is not a T5-TC (TrackCandidate.h:568), is skipped by CrossCleanT5 (224)
and protected in RemoveDupQuintupletsBeforeTC (Kernels.h:548); its pLS is skipped by the pT3 builder
(PixelTriplet.h CreatePixelTripletsFromMap 'if partOfPT5 continue'). For core failures (100 evt), count:
 T5-revivable: a genuine T5 with partOfPT5=1, AfterBuild-clean, every pT5 using it dedup-killed, and no SURVIVING pT5/pT3-TC
               sharing >=4 of its OT hits (so CrossCleanT5, TrackCandidate.h:258-271, would not kill it once un-flagged).
 pLS-freed:   a genuine unflagged pLS whose pT5s are all dedup-killed (pT3 builder skipped it) and that has a genuine T3."""
import collections, numpy as np, uproot
from load47 import ev, rows
from pair_attrib import bucket
F = '/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/Ntuple-files/LSTNtuple_s45_rebased_100evt.root'
X = uproot.open(F)['tree'].arrays(['t5_hitIndices'], library='np')
c = collections.Counter(); ev_all = collections.Counter()
for r in rows:
    ie = r['ev']; e = ev[ie]
    alive = e['pT5_isDupReco'] == 0
    survT5 = [int(k) for k, a in zip(e['pT5_t5Idx'], alive) if a]
    survHits = [set(np.asarray(X['t5_hitIndices'][ie][k]).tolist()) for k in survT5]
    fail = r['tc'] < 0
    for q in r['t5']:
        if not e['t5_partOfPT5'][q] or (e['t5_isDupBits'][q] & 1):
            continue
        users = [i for i, k in enumerate(e['pT5_t5Idx']) if k == q]
        if users and not any(alive[i] for i in users):
            h = set(np.asarray(X['t5_hitIndices'][ie][q]).tolist())
            blocked = any(len(h & sh) >= 4 for sh in survHits)
            key = 'T5 stale partOfPT5' + (' (blocked: surviving pT5 shares>=4 OT hits)' if blocked else ' (REVIVABLE)')
            ev_all[(key, 'fail' if fail else 'pass')] += 1
            if fail:
                c[(r['sim'], ie, key)] = 1
    pls_ok = [p for p in r['pls'] if not (e['pLS_isDup'][p] & 1)]
    for p in pls_ok:
        users = [i for i, pp in enumerate(e['pT5_plsIdx']) if pp == p]
        if users and not any(alive[i] for i in users):
            if fail:
                c[(r['sim'], ie, 'pLS stale partOfPT5' + (' + genuine T3' if r['t3'] else ''))] = 1
per = collections.Counter(k[2] for k in c)
print('distinct core FAILURES affected (100 evt):')
for k, v in sorted(per.items()): print(f'  {v:3d} {k}')
print('genuine-T5 instances (fail/pass sims):')
for k, v in sorted(ev_all.items()): print('  ', k, v)
# overlap with REVIVABLE per failure
rev = {(k[0], k[1]) for k in c if 'REVIVABLE' in k[2]}
print('distinct failures with >=1 revivable genuine T5:', len(rev), ' pT>100:', sum(1 for r in rows if (r['sim'], r['ev']) in rev and r['pt'] > 100))
print('their funnel buckets:', collections.Counter(bucket(r) for r in rows if (r['sim'], r['ev']) in rev))
