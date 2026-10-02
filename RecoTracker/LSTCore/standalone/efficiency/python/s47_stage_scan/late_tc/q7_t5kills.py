#!/usr/bin/env python3
"""Q3d: for core failing sims whose best genuine T5 was removed by AfterBuild (AB, Kernels.h:189-250), BeforeTC (BTC,
Kernels.h:498-586) or CrossCleanT5 (CC, TrackCandidate.h:207-276): who is the winner/killer and what happened to it?
AB winners replayed per module (60%-of-shorter rule, more layers wins, else higher dnn; ties -> lower index loses);
BTC exact replay; CC killers = alive pT5 (>= 4 shared OT hits, 0.15 window) or pT3 TC."""
import sys, collections
import numpy as np
from common import *


def main():
    C = collections.Counter()
    for ie, ev in events():
        H = Hits(ev)
        n5 = len(ev['t5_eta'])
        bits = ev['t5_isDupBits'].astype(int)
        part = ev['t5_partOfPT5'].astype(bool)
        eta = ev['t5_eta'].astype(np.float32); phi = ev['t5_phi'].astype(np.float32)
        dnn = ev['t5_dnnScore'].astype(np.float32); nl = ev['t5_nLayers'].astype(int)
        mod = ev['t5_moduleIdx'].astype(int)
        emb = np.array([np.asarray(x) for x in ev['t5_embed']]) if n5 else np.zeros((0, 6))
        pdup = ev['pT5_isDupReco'].astype(bool); p_t5 = ev['pT5_t5Idx'].astype(int); p_pls = ev['pT5_plsIdx'].astype(int)
        tc_type = ev['tc_type'].astype(int)
        t5tc = set(int(x) for x, t in zip(ev['tc_t5Idx'], tc_type) if t == 4)
        den = denominator(ev, 0.02)
        hitc = {}

        def hits(t):
            if t not in hitc:
                hitc[t] = H.t5_hits(t)
            return hitc[t]

        def fate(j, s):
            f = []
            if bits[j] & 1: f.append('AB')
            if bits[j] & 2: f.append('BTC')
            if bits[j] & 4: f.append('CC')
            if part[j]:
                al = [k for k in np.nonzero(p_t5 == j)[0] if not pdup[k]]
                if al:
                    gp = any(s in set(int(x) for x, fr in zip(ev['pT5_simIdxAll'][k], ev['pT5_simIdxAllFrac'][k]) if fr > .75)
                             for k in al)
                    f.append('in alive pT5 (%s)' % ('genuine' if gp else 'NOT genuine: wrong pLS'))
                else:
                    f.append('stale')
            if j in t5tc: f.append('T5 TC')
            return '+'.join(f) or 'alive?'

        for s in np.nonzero(den)[0]:
            if ev['sim_tcIdx'][s] >= 0:
                continue
            g = [int(x) for x in genuine(ev, 't5', s)]
            if not g:
                continue
            # furthest fate as in q3_fates
            best = None
            for f in ['isTC', 'in-alive-pT5', 'stale', 'CC', 'BTC', 'AB']:
                for t in g:
                    ok = {'isTC': t in t5tc,
                          'in-alive-pT5': part[t] and any(not pdup[k] for k in np.nonzero(p_t5 == t)[0]),
                          'stale': part[t] and not any(not pdup[k] for k in np.nonzero(p_t5 == t)[0]),
                          'CC': bool(bits[t] & 4) and not part[t], 'BTC': bool(bits[t] & 2) and not part[t],
                          'AB': bool(bits[t] & 1)}[f]
                    if ok:
                        best = (f, t)
                        break
                if best:
                    break
            if not best:
                continue
            f, t = best
            C['fate ' + f] += 1
            gset = set(g)
            if f == 'AB':
                ks = []
                for j in np.nonzero((mod == mod[t]) & (np.arange(n5) != t))[0]:
                    if abs(eta[j] - eta[t]) > 0.1 or abs(dphi(phi[j], phi[t])) > 0.1:
                        continue
                    a, b = (t, j) if t < j else (j, t)
                    nm = count_in(hits(a), set(hits(b)))
                    if nm < int(0.6 * min(2 * nl[a], 2 * nl[b])):
                        continue
                    if nl[a] > nl[b]: lose = b
                    elif nl[b] > nl[a]: lose = a
                    elif dnn[a] <= dnn[b]: lose = a
                    else: lose = b
                    if lose == t:
                        ks.append(int(j))
                if not ks:
                    C['  AB: no replayed winner (order race)'] += 1
                else:
                    kg = [j for j in ks if j in gset]
                    C['  AB: winner genuine same sim' if kg else '  AB: winner NOT genuine for sim'] += 1
                    for j in (kg or ks)[:1]:
                        C['    winner fate: ' + fate(j, s)] += 1
            elif f == 'BTC':
                ks = []
                for j in range(n5):
                    if j == t or bits[j] & 1 or (part[t] and part[j]):
                        continue
                    if abs(eta[j] - eta[t]) > 0.1 or abs(dphi(phi[j], phi[t])) > 0.1:
                        continue
                    if not (dnn[t] < dnn[j] or (dnn[t] == dnn[j] and t < j)):
                        continue
                    n1 = count_in(hits(t), set(hits(j))); n2 = count_in(hits(j), set(hits(t)))
                    d2 = float(np.sum((emb[t] - emb[j]) ** 2))
                    if (n1 >= 5 and d2 < .25) or n1 >= 10 or (n2 >= 5 and d2 < .25) or n2 >= 10:
                        ks.append(j)
                kg = [j for j in ks if j in gset]
                C['  BTC: killer genuine same sim' if kg else ('  BTC: killer NOT genuine' if ks else '  BTC: none?')] += 1
                for j in (kg or ks)[:1]:
                    C['    killer fate: ' + fate(j, s)] += 1
            elif f == 'CC':
                hs = set(hits(t))
                ks = []
                for k in np.nonzero(~pdup)[0]:
                    tt = p_t5[k]
                    if abs(eta[tt] - eta[t]) < .15 and abs(dphi(phi[tt], phi[t])) < .15 and len(hs & set(hits(tt))) >= 4:
                        ks.append(k)
                if ks:
                    k = ks[0]
                    same_t5 = any(p_t5[k] == t for k in ks)
                    plsg = any(int(p_pls[k]) in set(int(x) for x in genuine(ev, 'pls', s)) for k in ks)
                    C['  CC by alive pT5: pLS genuine for sim=%d' % plsg] += 1
                else:
                    C['  CC by pT3'] += 1
            elif f == 'in-alive-pT5':
                k = [k for k in np.nonzero(p_t5 == t)[0] if not pdup[k]][0]
                plsg = int(p_pls[k]) in set(int(x) for x in genuine(ev, 'pls', s))
                C['  alive pT5 uses genuine T5; pLS genuine=%d, sim has genuine pLS=%d, score>50=%d' % (
                    plsg, bool(len(genuine(ev, 'pls', s))), ev['pT5_score'][k] > 50)] += 1
        print(ie, file=sys.stderr, flush=True)
    for k, v in C.items():
        print(f'{v:>5}  {k}')


if __name__ == '__main__':
    main()
