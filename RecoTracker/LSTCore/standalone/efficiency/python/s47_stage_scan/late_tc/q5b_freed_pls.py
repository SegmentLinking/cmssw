#!/usr/bin/env python3
"""Upper bound on the side effect of pT5->T5 demotion: the demoted pT5's pLS is no longer pixel-overlap-killed by
CrossCleanpLS (TrackCandidate.h:350-357) and may appear as a pLS TC (only quad pLS, AddpLSasTrackCandidate:614).
The ntuple's final pLS_isDup includes CrossCleanpLS (writes =true), so this counts ALL quad pLS of demoted pT5s
(upper bound; the embedding check vs the demoted T5, TrackCandidate.h:325-338, may still kill them)."""
import pickle, collections
from common import CACHE
E = pickle.load(open(CACHE + '/../q5_tcs_v2.pkl', 'rb'))
for thr in [20, 50, 100, 200]:
    c = collections.Counter()
    for e in E:
        found = collections.Counter()
        for d in e['tcs']:
            ss = d['tsims'] if (d['type'] == 7 and d['score'] > thr) else d['sims']
            for s in ss: found[s] += 1
        dens = {sd['s']: sd for sd in e['sims']}
        for d in e['tcs']:
            if d['type'] != 7 or d['score'] <= thr or not d['quad']: continue
            c['quad pLS freed'] += 1
            if not d['psims']:
                c['  fake pLS'] += 1; continue
            s = next(iter(d['psims']))
            if found[s]: c['  genuine, sim already has a TC (dup)'] += 1
            else:
                c['  genuine, sim has no TC (potential gain)'] += 1
                if s in dens and dens[s]['core']: c['    ... core-den sim'] += 1
                elif s in dens: c['    ... dR<0.1-den sim'] += 1
            if d['plsdup'] & 2: c['  (has CheckHitspLS TC-stage bit -> stays dead)'] += 1
    print(f'score > {thr}:', dict(c))
