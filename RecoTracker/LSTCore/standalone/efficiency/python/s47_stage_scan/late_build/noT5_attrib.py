#!/usr/bin/env python3
"""'no_T5' bucket (genuine unflagged pLS, no genuine T5) on the S45 100-evt ntuple:
does a pair of genuine T3s exist that CreateQuintuplets would have tried (inner T3 starts in a valid T5 region
[barrel L1/L2 or endcap L1, Quintuplet.h:2063-2068] and its outer MD == outer T3's inner MD, Quintuplet.h:1951-1960)?
If yes, the T5 was rejected at creation (dBeta x2, r-z, or the new T5 DNN: Quintuplet.h:1600-1690) or truncated;
the ntuple keeps only CREATED T5s, so which of those cuts fired needs instrumentation.
Also: best partially-matched T5 fraction, genuine T4/pT3 availability."""
import collections, numpy as np
from load47 import ev, rows
from pair_attrib import bucket


def t3md(e, t):
    return (int(e['ls_mdIdx0'][e['t3_lsIdx0'][t]]), int(e['ls_mdIdx1'][e['t3_lsIdx0'][t]]), int(e['ls_mdIdx1'][e['t3_lsIdx1'][t]]))


def valid_start(e, t):
    L = e['t3_hit_0_layer'][t]
    return L in (1, 2, 7)


c = collections.Counter(); det = []
for r in rows:
    if r['tc'] >= 0 or bucket(r) != 'no_T5':
        continue
    e = ev[r['ev']]; s = r['sim']
    g3 = r['t3']
    mds = {t: t3md(e, t) for t in g3}
    pairs = [(a, b) for a in g3 for b in g3 if a != b and mds[a][2] == mds[b][0] and valid_start(e, a)]
    fr = np.asarray(e['sim_t5IdxAllFrac'][s]); bestfrac = float(fr.max()) if len(fr) else 0.
    hi = r['pt'] > 100
    if not g3:
        k = 'a no genuine T3 at all (T3/LS/MD lane)'
    elif not any(valid_start(e, t) for t in g3):
        k = 'b genuine T3s, none starts in a T5 seed region (B1/B2/E1)'
    elif not pairs:
        k = 'c genuine T3 in seed region but no genuine outer T3 sharing its outer MD'
    else:
        k = 'd genuine T3 pair exists -> rejected at T5 creation (dBeta/rz/T5-DNN) [needs instrumentation]'
    c[k] += 1; c[(k, 'pT>100')] += hi
    c[(k, 'T5 frac 0.5-0.75 exists')] += (0.5 <= bestfrac < 0.75)
    c[(k, 'genuine pT3')] += bool(r['pt3']); c[(k, 'genuine T4')] += bool(r['t4'])
    c[(k, 'genuine T3 partOfPT3')] += any(e['t3_partOfPT3'][t] for t in g3)
    det.append((k[0], r['pt'], len(g3), len(pairs), bestfrac, bool(r['pt3']), bool(r['t4'])))
print('no_T5 bucket, N=%d' % sum(v for k, v in c.items() if isinstance(k, str)))
for k in sorted(x for x in c if isinstance(x, str)):
    print(f'  {c[k]:3d} {k}   [pT>100 {c[(k, "pT>100")]}, has T5 w/ frac 0.5-0.75: {c[(k, "T5 frac 0.5-0.75 exists")]}, '
          f'genuine pT3: {c[(k, "genuine pT3")]}, genuine T4: {c[(k, "genuine T4")]}, gen T3 in a pT3: {c[(k, "genuine T3 partOfPT3")]}]')
print('\ndetail (cat, simPt, nGenT3, nGenT3pairs, best T5 frac, genPT3, genT4):')
for d in sorted(det):
    print('  %s %7.1f %3d %3d %.2f %d %d' % d)
