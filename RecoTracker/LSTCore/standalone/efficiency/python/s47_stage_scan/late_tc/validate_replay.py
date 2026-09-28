#!/usr/bin/env python3
"""Check (1) replay of RemoveDupPixelQuintupletsFromMap reproduces pT5_isDupReco; (2) t5_t3Idx0 == t5_t3_idx0;
(3) T5 base hitIndices <-> MD-coordinate mapping is one-to-one."""
import sys, collections
import numpy as np
from common import *

c = collections.Counter()
for ie, ev in events():
    H = Hits(ev)
    c['t3idx_same'] += int(np.all(ev['t5_t3Idx0'] == ev['t5_t3_idx0']) and np.all(ev['t5_t3Idx1'] == ev['t5_t3_idx1']))
    c['ev'] += 1
    # coord<->hidx injectivity
    c['h2k'] += len(H.h2k); c['k2h'] += len(H.k2h)
    n = len(ev['pT5_score'])
    hits, nbr, nmat = replay_pt5_dedup(ev, H)
    sc = ev['pT5_score']
    rep = np.array([any((sc[i] > sc[j]) or (sc[i] == sc[j] and i > j) for j in nbr[i]) for i in range(n)], bool)
    true = ev['pT5_isDupReco'].astype(bool)
    c['pt5'] += n; c['agree'] += int((rep == true).sum()); c['rep1_true0'] += int((rep & ~true).sum())
    c['rep0_true1'] += int((~rep & true).sum())
    # unmapped extended hits
    for t in range(len(ev['t5_nLayers'])):
        for h in ev['t5_hitIndices'][t]:
            if int(h) not in H.h2k: c['unmapped_t5_hit'] += 1
    if ie % 20 == 0: print(ie, dict(c), file=sys.stderr, flush=True)
print(dict(c))
