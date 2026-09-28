#!/usr/bin/env python3
"""Q3/Q4: fates of genuine objects of core (dR<0.02) failing tracks on the current code.

Per failing sim: genuine (frac >= 0.75) pLS / T5 / T4 / pT3 / pT5 and why none became a matched TC.
T5 bits (Kernels.h:22-26, TrackCandidate.h:268): isDup |= 1 AfterBuild, |= 2 BeforeTC, |= 4 CrossCleanT5.
T5 with partOfPT5 is never a TC (TrackCandidate.h:568 AddT5asTrackCandidate, :474 count, :224 CrossCleanT5 skip).
pLS bits: bit0 CheckHitspLS pre-pT5, bit1 CheckHitspLS TC stage; CrossCleanpLS writes isDup = true (=1, overwrites).
T4: isDup |= 1 AfterBuild, |= 2 BeforeTC, CrossCleanT4 writes = true (1).  (TrackCandidate.h:421/426)
Near-miss: best TC match fraction for the failing sim (sim_tcIdxAllFrac)."""
import sys, collections, json
import numpy as np
from common import *


def main():
    C = collections.Counter()
    near = collections.Counter()
    near_rows = []
    t5fate = collections.Counter()
    t4fate = collections.Counter()
    plsfate = collections.Counter()
    pt3fate = collections.Counter()
    stale = collections.Counter()
    rows = []
    for ie, ev in events():
        den = denominator(ev, 0.02)
        simtc = ev['sim_tcIdx']
        tc_type = ev['tc_type'].astype(int)
        pt5dup = ev['pT5_isDupReco'].astype(bool)
        p_pls = ev['pT5_plsIdx'].astype(int)
        p_t5 = ev['pT5_t5Idx'].astype(int)
        t5_pt5s = collections.defaultdict(list)
        pls_pt5s = collections.defaultdict(list)
        for i in range(len(p_t5)):
            t5_pt5s[int(p_t5[i])].append(i)
            pls_pt5s[int(p_pls[i])].append(i)
        pt3_in_tc = set(int(x) for x, ty in zip(ev['tc_pt3Idx'], tc_type) if ty == 5)
        pls_in_tc = set(int(x) for x, ty in zip(ev['tc_plsIdx'], tc_type) if ty == 8)
        t4_in_tc = set(int(x) for x, ty in zip(ev['tc_t4Idx'], tc_type) if ty == 9)
        pt3_pls = ev['pT3_plsIdx'].astype(int)
        # --- Q4 stale partOfPT5 (all T5s / pLS, event-wide)
        t5p5 = ev['t5_partOfPT5'].astype(bool)
        for t in np.nonzero(t5p5)[0]:
            alive = [i for i in t5_pt5s[int(t)] if not pt5dup[i]]
            stale['T5 partOfPT5'] += 1
            if not alive:
                stale['T5 partOfPT5 but all its pT5 dead (stale)'] += 1
                if not (ev['t5_isFake'][t]):
                    stale['  ... stale and T5 genuine(not isFake)'] += 1
        for p in pls_pt5s:
            stale['pLS partOfPT5'] += 1
            if all(pt5dup[i] for i in pls_pt5s[p]):
                stale['pLS partOfPT5 but all its pT5 dead (stale)'] += 1
                if not ev['pLS_isFake'][p]:
                    stale['  ... stale and pLS genuine'] += 1

        for s in np.nonzero(den)[0]:
            C['den'] += 1
            if simtc[s] >= 0:
                continue
            C['fail'] += 1
            g = {o: [int(x) for x in genuine(ev, o, s)] for o in ['pls', 't5', 'pt5', 'pt3', 't4', 't3']}
            fr = np.asarray(ev['sim_tcIdxAllFrac'][s])
            ti = np.asarray(ev['sim_tcIdxAll'][s]).astype(int)
            best = float(fr.max()) if len(fr) else 0.0
            bk = int(ti[np.argmax(fr)]) if len(fr) else -1
            b = 'none' if best == 0 else ('<0.5' if best < 0.5 else ('0.5-0.6' if best < 0.6 else (
                '0.6-0.75' if best <= 0.75 else '>0.75?')))
            near[b] += 1
            key = tuple(o for o in ['pls', 't5', 'pt5', 'pt3', 't4'] if g[o])
            C['has:' + ('+'.join(key) or 'nothing')] += 1
            if best >= 0.6 and bk >= 0:
                ty = int(tc_type[bk])
                info = dict(ev=ie, s=int(s), best=best, type=ty, nhits=int(ev['tc_nhits'][bk]))
                if ty == 7:
                    i = int(ev['tc_pt5Idx'][bk])
                    info['pls_gen'] = int(p_pls[i]) in g['pls']
                    info['t5_gen'] = int(p_t5[i]) in g['t5']
                    info['t5_nl'] = int(ev['t5_nLayers'][p_t5[i]])
                elif ty == 4:
                    t = int(ev['tc_t5Idx'][bk])
                    info['t5_nl'] = int(ev['t5_nLayers'][t])
                    info['t5_base_frac'] = None
                near_rows.append(info)
                nm = {7: 'pT5', 4: 'T5', 5: 'pT3', 8: 'pLS', 9: 'T4'}[ty]
                tag = nm
                if ty == 7:
                    tag += ' pLS_gen=%d T5_gen=%d' % (info['pls_gen'], info['t5_gen'])
                near['nearmiss>=0.6 ' + tag] += 1
            # ---- T5 fates (only for failing sims)
            for t in g['t5']:
                bits = int(ev['t5_isDupBits'][t])
                f = []
                if bits & 1: f.append('AB')
                if bits & 2: f.append('BTC')
                if bits & 4: f.append('CC')
                if t5p5[t]:
                    al = [i for i in t5_pt5s[t] if not pt5dup[i]]
                    f.append('pT5-alive' if al else 'partOfPT5-stale')
                if ev['t5_partOfTC'][t]:
                    f.append('isTC(nl=%d)' % int(ev['t5_nLayers'][t]))
                t5fate['/'.join(f) or 'none?'] += 1
            if g['t5']:
                # per-sim: best fate
                fs = set()
                for t in g['t5']:
                    bits = int(ev['t5_isDupBits'][t])
                    if ev['t5_partOfTC'][t]: fs.add('isTC')
                    elif t5p5[t] and any(not pt5dup[i] for i in t5_pt5s[t]): fs.add('in-alive-pT5')
                    elif t5p5[t]: fs.add('stale')
                    elif bits & 4: fs.add('CC')
                    elif bits & 2: fs.add('BTC')
                    elif bits & 1: fs.add('AB')
                for f in ['isTC', 'in-alive-pT5', 'stale', 'CC', 'BTC', 'AB']:
                    if f in fs:
                        C['simT5best:' + f] += 1
                        break
            for t in g['t4']:
                t4fate['TC' if t in t4_in_tc else 'noTC (t4_isDup not filled in this build)'] += 1
            if g['t4']:
                C['simT4:' + ('someTC' if any(t in t4_in_tc for t in g['t4']) else 'noTC')] += 1
            for p in g['pls']:
                d = int(ev['pLS_isDup'][p])
                pf = 'quad' if ev['pLS_isQuad'][p] else 'trip'
                pf += ' isDup=%d' % d
                if p in pls_pt5s:
                    pf += ' inPT5(' + ('alive' if any(not pt5dup[i] for i in pls_pt5s[p]) else 'stale') + ')'
                if p in set(pt3_pls.tolist()):
                    pf += ' inPT3'
                if p in pls_in_tc:
                    pf += ' TC'
                plsfate[pf] += 1
            for q in g['pt3']:
                pt3fate['TC' if q in pt3_in_tc else 'noTC'] += 1
            rows.append(dict(ev=ie, s=int(s), pt=float(ev['sim_pt'][s]), best=best, g={k: v for k, v in g.items()}))
    out = []
    out.append('== counts: ' + json.dumps({k: v for k, v in C.items() if not k.startswith(('has', 'sim'))}))
    for title, cnt in [('genuine-object content of failing core sims', {k: v for k, v in C.items() if k.startswith('has')}),
                       ('per failing sim with genuine T5: furthest T5 fate', {k: v for k, v in C.items() if k.startswith('simT5')}),
                       ('per failing sim with genuine T4', {k: v for k, v in C.items() if k.startswith('simT4')}),
                       ('genuine T5 objects of failing sims: fate', t5fate), ('genuine T4 objects: fate', t4fate),
                       ('genuine pLS objects: fate', plsfate), ('genuine pT3 objects', pt3fate),
                       ('best TC match fraction of failing core sims', near), ('Q4 stale partOfPT5 (event-wide)', stale)]:
        out.append('-- ' + title)
        for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
            out.append(f'   {v:>6}  {k}')
    print('\n'.join(out))
    json.dump(dict(rows=rows, near=near_rows), open('q3_rows.json', 'w'), default=float)


if __name__ == '__main__':
    main()
