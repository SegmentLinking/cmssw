#!/usr/bin/env python3
"""Core funnel (s46_core_funnel.py buckets) + kernel-order waterfall for the 'pair_failed' bucket
(genuine unflagged pLS and genuine T5 exist, no genuine pT5) on the S45 rebased 100-evt ntuple.
Emulated exactly: superbin validity + pixel-map connectivity + 2S inner veto (PixelQuintuplet.h:704-711),
pLS isDup (706), T5 AfterBuild isDup (722), passRadiusCriterion (PixelTriplet.h:590 -> 450-466, bounds 350-445).
Not emulated: PPBB/PPEE tracklet x2 (PixelTriplet.h:600-620), pT3-DNN pT5WP (688), pT5 chi2 cuts
(PixelQuintuplet.h:604/633/648; only if pixel radius / T5 radius < 5 GeV)."""
import struct, collections, numpy as np, pickle, sys
from load47 import ev, rows, plsinfo, SCR
kR1GeVf = 1. / (2.99792458e-3 * 3.8)
MD = '/cvmfs/cms.cern.ch/el8_amd64_gcc13/cms/cmssw/CMSSW_16_1_1/external/el8_amd64_gcc13/data/RecoTracker/LSTCore/data/OT800_IT615_pt0.8/pixelmap/'


def loadmap(fn):
    d = {}; b = open(fn, 'rb').read(); o = 0
    while o + 8 <= len(b):
        k, n = struct.unpack_from('II', b, o); o += 8
        d[k] = set(struct.unpack_from('%dI' % n, b, o)); o += 4 * n
    return d


L = ['_layer1_subdet5', '_layer2_subdet5', '_layer1_subdet4', '_layer2_subdet4']
maps = {0: [loadmap(MD + 'pLS_map' + x + '.bin') for x in L], 1: [loadmap(MD + 'pLS_map_pos' + x + '.bin') for x in L],
        2: [loadmap(MD + 'pLS_map_neg' + x + '.bin') for x in L]}


def connected(pi, detids):
    sb = pi['sb']
    if sb < 0 or sb >= 45000:
        return None
    key = sb + 45000 if pi['ptype'] == 0 else sb
    s = set()
    for m in maps[pi['ptype']]:
        s |= m.get(key, set())
    return any(d in s for d in detids)


BOUNDS = {'BBB': ((0.15624, 0.17235), (0.6588, 0.6375)), 'BBE': ((0.45972, 0.19644), (0.8557, 0.6805)),
          'BEE': ((1.59294, 0.255181), (2.3548, 2.2091)), 'EEE': ((1.7006, 0.26367), (2.436, 2.286))}


def radius_pass(pR, pRe, tR, kind):
    (tb, pb), (tbh, pbh) = BOUNDS[kind]
    if pR > 2 * kR1GeVf:
        tb, pb = tbh, pbh
    tmax = (1 + tb) / tR; tmin = max((1 - tb) / tR, 0.)
    pmax = max((1 + pb) / pR, 1. / (pR - pRe)) if pR > pRe else 1e9
    pmin = min((1 - pb) / pR, 1. / (pR + pRe))
    if kind in ('BEE', 'EEE'):
        pmin = max(pmin, 0.)
    return (tmin <= pmin < tmax) or (pmin < tmin < pmax)


def t3kind(e, i):
    E = [e['t3_hit_%d_layer' % k][i] > 6 for k in (0, 2, 4)]
    return 'EEE' if E[0] else 'BEE' if E[1] else 'BBE' if E[2] else 'BBB'


def t5dets(e, q):
    it3 = e['t5_t3Idx0'][q]
    return [int(e['t3_hit_0_detId'][it3]), int(e['t3_hit_1_detId'][it3])]


STAGES = ['1 superbin invalid', '2 map not connected / inner 2S', '3 T5 AfterBuild-isDup', '4 radius criterion',
          '5 passes emulated cuts -> tracklet PPBB/PPEE, pT3-DNN(pT5WP) or chi2']


def pairstage(ie, p, q):
    e = ev[ie]; pi = plsinfo[ie][p]
    if pi is None or not (0 <= pi['sb'] < 45000):
        return 0
    it3 = e['t5_t3Idx0'][q]
    if e['t3_hit_0_moduleType'][it3] != 0 or not connected(pi, t5dets(e, q)):
        return 1
    if e['t5_isDupBits'][q] & 1:
        return 2
    if not radius_pass(e['pLS_pt'][p] * kR1GeVf, e['pLS_ptErr'][p] * kR1GeVf, e['t3_radius'][it3], t3kind(e, it3)):
        return 3
    return 4


def bucket(r):
    e = ev[r['ev']]
    if r['pt5']:
        return 'pT5_survived' if any(e['pT5_isDupReco'][i] == 0 for i in r['pt5']) else 'pT5_all_killed'
    if not r['pls']:
        return 'no_pLS'
    if not [p for p in r['pls'] if not (e['pLS_isDup'][p] & 1)]:
        return 'pLS_flagged'
    if not r['t5']:
        return 'no_T5'
    return 'pair_failed'


if __name__ == '__main__':
    # validation: built pT5 pairs must reach stage 5 (connectivity/radius emulation)
    v = collections.Counter()
    for ie, e in enumerate(ev):
        for p, q in zip(e['pT5_plsIdx'], e['pT5_t5Idx']):
            v[STAGES[pairstage(ie, p, q)]] += 1
    print('VALIDATION on all built pT5 (pLS,T5) pairs:', dict(v))
    fails = [r for r in rows if r['tc'] < 0]
    B = collections.Counter(bucket(r) for r in fails)
    print('core funnel (100 evt, den %d, fails %d):' % (len(rows), len(fails)), dict(B))
    PF = [r for r in fails if bucket(r) == 'pair_failed']
    c = collections.Counter(); ex = collections.defaultdict(list)
    for r in PF:
        e = ev[r['ev']]
        pls = [p for p in r['pls'] if not (e['pLS_isDup'][p] & 1)]
        best = max((pairstage(r['ev'], p, q), p, q) for p in pls for q in r['t5'])
        k = STAGES[best[0]]
        c[k] += 1
        # annotations
        stolenT5 = any(e['t5_partOfPT5'][q] for q in r['t5'])
        pls_in_pt5 = [p for p in pls if p in set(e['pT5_plsIdx'])]
        t5tc = any(e['t5_partOfTC'][q] for q in r['t5'])
        _, p, q = best
        ex[k].append(dict(pt=r['pt'], plspt=float(e['pLS_pt'][p]), plsptErr=float(e['pLS_ptErr'][p]), quad=int(e['pLS_isQuad'][p]),
                          t5dup=int(e['t5_isDupBits'][q]), stolenT5=stolenT5, pls_in_foreign_pT5=bool(pls_in_pt5), t5_partOfTC=t5tc,
                          pT3=bool(r['pt3']), T4=bool(r['t4']), kind=t3kind(e, e['t5_t3Idx0'][q]), t3R=float(e['t3_radius'][e['t5_t3Idx0'][q]])))
    print('\npair_failed waterfall (furthest stage reached by any genuine unflagged-pLS x genuine-T5 pair): N=%d' % len(PF))
    for k in STAGES:
        if c[k]:
            d = ex[k]
            print(f'  {c[k]:3d}  {k:62s} simPt med {np.median([x["pt"] for x in d]):6.0f}  pT>100: {sum(x["pt"] > 100 for x in d)}')
    print('\nper-track detail:')
    for k in STAGES:
        for x in ex[k]:
            print(f'  [{k[0]}] simPt {x["pt"]:7.1f} pLSpt {x["plspt"]:7.1f}+-{x["plsptErr"]:6.1f} quad {x["quad"]} T3kind {x["kind"]} '
                  f't3R->pt {x["t3R"] / kR1GeVf:7.1f} T5dupBits {x["t5dup"]} T5stolen(partOfPT5) {int(x["stolenT5"])} '
                  f'pLS-in-foreign-pT5 {int(x["pls_in_foreign_pT5"])} T5isTC {int(x["t5_partOfTC"])} gen pT3 {int(x["pT3"])} gen T4 {int(x["T4"])}')
    pickle.dump(dict(ex), open(SCR + 'pair_attrib.pkl', 'wb'))
