#!/usr/bin/env python3
"""Core failures with a genuine pT3 removed by CrossCleanpT3 (TrackCandidate.h:172-204): which pT5 pLS matched it
(dead or alive pT5; that pT5 genuine for this sim or not)? CrossCleanpT3 loops over ALL pT5s without an isDup check."""
import numpy as np, collections
from load47 import ev, rows
c = collections.Counter()
for r in rows:
    if r['tc'] >= 0 or not r['pt3']:
        continue
    e = ev[r['ev']]
    for i in r['pt3'][:1]:
        p = e['pT3_plsIdx'][i]; eta1, phi1 = e['pLS_eta'][p], e['pLS_phi'][p]
        hits = [j for j, q in enumerate(e['pT5_plsIdx']) if (eta1 - e['pLS_eta'][q]) ** 2 + ((phi1 - e['pLS_phi'][q] + np.pi) % (2 * np.pi) - np.pi) ** 2 < 1e-5]
        if not hits:
            c['not CrossCleanpT3'] += 1; continue
        alive = [j for j in hits if e['pT5_isDupReco'][j] == 0]
        gen = [j for j in hits if e['pT5_simIdx'][j] == r['sim']]
        c['killer pT5s: ' + ('some alive' if alive else 'ALL dedup-killed') + (', genuine for sim' if gen else ', foreign/fake')] += 1
for k, v in c.items(): print(v, k)
