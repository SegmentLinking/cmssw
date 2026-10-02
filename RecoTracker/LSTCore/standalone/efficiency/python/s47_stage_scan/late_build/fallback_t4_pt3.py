#!/usr/bin/env python3
"""Q3: core failures (100 evt) with a genuine T4 or pT3 that did not become a TC, and why.
pT3: skipped for pLS with partOfPT5 (PixelTriplet.h CreatePixelTripletsFromMap 'if partOfPT5 continue');
killed by RemoveDupPixelTripletsFromMap (Kernels.h:705) or CrossCleanpT3 (TrackCandidate.h:172-204: pLS eta/phi within
dR2<1e-5 of the pLS of ANY pT5, dup-killed or not). T4: created only if P(displaced) > WP (NeuralNetwork.h:570),
then RemoveDupQuadruplets*, CrossCleanT4 (TrackCandidate.h:391-434)."""
import collections, numpy as np
from load47 import ev, rows
from pair_attrib import bucket
c = collections.Counter()
for r in rows:
    if r['tc'] >= 0:
        continue
    e = ev[r['ev']]; b = bucket(r)
    c['fails'] += 1
    if r['pt3']:
        c['fail w/ genuine pT3'] += 1; c[('pT3', b)] += 1
        tcpt3 = set(e['tc_pt3Idx'])
        pt5pls = [int(p) for p in e['pT5_plsIdx']]
        for i in r['pt3'][:1]:
            p = e['pT3_plsIdx'][i]
            if i in tcpt3:
                why = 'is a TC (?)'
            else:
                eta1, phi1 = e['pLS_eta'][p], e['pLS_phi'][p]
                cc = any((eta1 - e['pLS_eta'][q]) ** 2 + ((phi1 - e['pLS_phi'][q] + np.pi) % (2 * np.pi) - np.pi) ** 2 < 1e-5 for q in pt5pls)
                dead = [q for q, k, d in zip(e['pT5_plsIdx'], e['pT5_t5Idx'], e['pT5_isDupReco']) if abs(e['pLS_eta'][q] - eta1) < 3e-3]
                why = 'CrossCleanpT3 (pLS ~same eta/phi as a pT5 pLS)' if cc else 'RemoveDupPixelTriplets (by elimination)'
            c[('pT3 why', why)] += 1
    if r['t4']:
        c['fail w/ genuine T4'] += 1; c[('T4', b)] += 1
        st = collections.Counter()
        for i in r['t4']:
            st['partOfTC' if e['t4_partOfTC'][i] else 'notTC (isDup branch empty in ntuple)'] += 1
        c[('T4 status', ' '.join(sorted(st)))] += 1
    if r['t3'] and not r['t5'] and not r['pt3'] and not r['t4']:
        c['fail: genuine T3 only (no T5/pT3/T4)'] += 1
print(f"core failures {c['fails']}; with genuine pT3 {c['fail w/ genuine pT3']}; with genuine T4 {c['fail w/ genuine T4']}; "
      f"genuine T3 but no genuine T5/pT3/T4 {c['fail: genuine T3 only (no T5/pT3/T4)']}")
for k in sorted((k for k in c if isinstance(k, tuple)), key=str):
    print('  ', k, c[k])
# core-wide: genuine T4 frequency among ALL core sims vs failures; displaced-score of genuine T4s
n4 = sum(bool(r['t4']) for r in rows); print(f'core sims with any genuine T4: {n4}/{len(rows)}')
