#!/usr/bin/env python3
"""Combined counterfactual of the late-stage fixes (first-order TC-set model with exact kernel replays):
  pT5 dedup  : RemoveDupPixelQuintupletsFromMap replay (Kernels.h:735-768), key/mode variants  (Fix C)
  demotion   : alive pT5 with score (rPhiChi2, FP16) > X emitted as T5 TC (AddpT5asTrackCandidate, TrackCandidate.h:638)
  unstale    : partOfPT5 of a T5 cleared when all its pT5s are dead (PixelQuintuplet.h:791 / Kernels.h:762)
T5 TCs = replayed RemoveDupQuintupletsBeforeTC (validated 4536/4536) + CrossCleanT5 (555/567) with the scenario's
alive pT5 set and partOfPT5 flags.  pT3/T4/pLS TCs are kept as in the ntuple (CrossCleanpLS / CrossCleanT4 /
freed-pLS effects ignored; see q5b_freed_pls.txt and q4_unstale.txt for their size).
Train = events 0-49, test = 50-99."""
import sys, collections, pickle, math, os
import numpy as np
from common import *
from q12_pt5_dedup import dead_parallel, dead_greedy

PK = CACHE + '/../q6_pre.pkl'


def precompute():
    out = []
    for ie, ev in events():
        H = Hits(ev)
        n5 = len(ev['t5_eta'])
        bits = ev['t5_isDupBits'].astype(int)
        abdup = (bits & 1).astype(bool)
        t5hits = [H.t5_hits(t) for t in range(n5)]
        t5sets = [set(h) for h in t5hits]
        eta = ev['t5_eta'].astype(np.float32); phi = ev['t5_phi'].astype(np.float32)
        dnn = ev['t5_dnnScore'].astype(np.float32)
        emb = np.array([np.asarray(x) for x in ev['t5_embed']]) if n5 else np.zeros((0, 6))
        cand = np.nonzero(~abdup)[0]
        losers = collections.defaultdict(list)  # i -> list of j that beat i in BTC (partOfPT5 exemption applied later)
        for i in cand:
            m = (np.abs(eta[cand] - eta[i]) <= 0.1) & (np.abs(dphi(phi[cand], phi[i])) <= 0.1) & (cand != i)
            for j in cand[m]:
                if not (dnn[i] < dnn[j] or (dnn[i] == dnn[j] and i < j)):
                    continue
                n1 = count_in(t5hits[i], t5sets[j]); n2 = count_in(t5hits[j], t5sets[i])
                d2 = float(np.sum((emb[i] - emb[j]) ** 2))
                isd = lambda n: (n >= 5 and d2 < 0.25) or n >= 10
                if isd(n1) or isd(n2):
                    losers[int(i)].append(int(j))
        # pT5
        hits, nbr, nmat = replay_pt5_dedup(ev, H)
        npt5 = len(hits)
        p_t5 = ev['pT5_t5Idx'].astype(int); p_pls = ev['pT5_plsIdx'].astype(int)
        pt5_eta = f16(ev['t5_eta'][p_t5]) if npt5 else np.zeros(0)
        pt5_phi = f16(ev['t5_phi'][p_t5]) if npt5 else np.zeros(0)
        # CrossCleanT5 relations for every non-AB T5 (partOfPT5 decided per scenario)
        tc_type = ev['tc_type'].astype(int)
        cc_pt3 = set()
        pt3s = []
        for k in np.nonzero(tc_type == 5)[0]:
            q = int(ev['tc_pt3Idx'][k])
            pt3s.append((float(ev['pT3_eta'][q]), float(ev['pT3_phi'][q]), set(H.t3_hits(int(ev['pT3_t3Idx'][q])))))
        cc_pt5 = collections.defaultdict(list)
        for i in cand:
            hs = t5sets[i]
            if npt5:
                m = np.nonzero((np.abs(pt5_eta - eta[i]) < 0.15) & (np.abs(dphi(pt5_phi, phi[i])) < 0.15))[0]
                for k in m:
                    if sum(1 for h in hs if h in set(t5hits[p_t5[k]])) >= 4:
                        cc_pt5[int(i)].append(int(k))
            for (e, p, ot) in pt3s:
                if abs(eta[i] - e) < 0.15 and abs(dphi(phi[i], p)) < 0.15 and sum(1 for h in hs if h in ot) >= 4:
                    cc_pt3.add(int(i)); break
        simset = lambda a, f: set(int(s) for s, x in zip(a, f) if x > TCMATCH)
        t5sims = [simset(ev['t5_simIdxAll'][t], ev['t5_simIdxAllFrac'][t]) for t in range(n5)]
        pt5sims = [simset(ev['pT5_simIdxAll'][k], ev['pT5_simIdxAllFrac'][k]) for k in range(npt5)]
        others = []  # non-pT5, non-T5 TCs: (sims, eta, phi)
        for k in range(len(tc_type)):
            if tc_type[k] in (5, 8, 9):
                others.append((simset(ev['tc_simIdxAll'][k], ev['tc_simIdxAllFrac'][k]), float(ev['tc_eta'][k]),
                               float(ev['tc_phi'][k])))
        sel_j = [(float(e), float(p)) for e, p, pt in zip(ev['genjet_eta'], ev['genjet_phi'], ev['genjet_pt'])
                 if pt > 1000 and abs(e) < 2.5]
        den02, den10 = denominator(ev, 0.02), denominator(ev, 0.10)
        sims = [(int(s), bool(den02[s]), bool(ev['sim_tcIdx'][s] >= 0)) for s in np.nonzero(den10)[0]]
        from q12_pt5_dedup import pix_resid
        out.append(dict(ie=ie, n5=n5, abdup=abdup, part=ev['t5_partOfPT5'].astype(bool), losers=dict(losers),
                        cc_pt5=dict(cc_pt5), cc_pt3=cc_pt3, t5sims=t5sims, t5eta=eta, t5phi=phi,
                        npt5=npt5, nbr=nbr, p_t5=p_t5, sc=ev['pT5_score'].astype(np.float32),
                        dup=ev['pT5_isDupReco'].astype(bool), pt5sims=pt5sims, gen=np.array([bool(x) for x in pt5sims]),
                        nl=ev['t5_nLayers'][p_t5].astype(int) if npt5 else np.zeros(0, int),
                        dnn=ev['t5_dnnScore'][p_t5].astype(float) if npt5 else np.zeros(0),
                        pe=ev['pT5_eta'].astype(float), pp=ev['pT5_phi'].astype(float),
                        others=others, sel_j=sel_j, sims=sims))
        print('pre', ie, file=sys.stderr, flush=True)
    return out


def jdr(e, eta, phi):
    return min([math.hypot(eta - a, float(dphi(phi, b))) for a, b in e['sel_j']] or [9.])


def scenario(e, key='score', mode='par', demote=None, unstale=False):
    n = e['npt5']
    if n:
        if key == 'score' and mode == 'par':
            dead = e['dup']
        else:
            k = {'score': e['sc'].astype(float),
                 'nl>score': -100.0 * e['nl'] + np.log(np.maximum(e['sc'], 1e-4)),
                 'oracle': 1000.0 * (~e['gen']) + np.log(np.maximum(e['sc'], 1e-4))}[key]
            dead = dead_parallel(k, e['nbr']) if mode == 'par' else dead_greedy(k, e['nbr'])
    else:
        dead = np.zeros(0, bool)
    alive = np.nonzero(~dead)[0]
    has_alive = np.zeros(e['n5'], bool)
    if len(alive):
        has_alive[e['p_t5'][alive]] = True
    # partOfPT5 as it would be on device: set for every T5 of a BUILT pT5 (all pT5 in ntuple), cleared if unstale
    part = e['part'] & has_alive if unstale else e['part']
    alive_set = set(alive.tolist())
    tcs = []  # (sims, eta, phi, kind)
    for i in range(e['n5']):
        if e['abdup'][i] or part[i]:
            continue
        if any(not (part[i] and part[j]) for j in e['losers'].get(i, [])):
            continue
        if i in e['cc_pt3'] or any(k in alive_set for k in e['cc_pt5'].get(i, [])):
            continue
        tcs.append((e['t5sims'][i], float(e['t5eta'][i]), float(e['t5phi'][i]), 'T5'))
    for k in alive:
        if demote is not None and e['sc'][k] > demote:
            t = e['p_t5'][k]
            tcs.append((e['t5sims'][t], float(e['t5eta'][t]), float(e['t5phi'][t]), 'dT5'))
        else:
            tcs.append((e['pt5sims'][k], float(e['pe'][k]), float(e['pp'][k]), 'pT5'))
    tcs += [(s, a, b, 'o') for (s, a, b) in e['others']]
    return tcs


def score_tcs(e, tcs):
    r = collections.Counter()
    found = set()
    nmatch = collections.Counter()
    for s, a, b, kind in tcs:
        found |= s
        for x in s:
            nmatch[x] += 1
        if kind == 'dT5':
            r['ndemoted'] += 1
        if not s:
            d = jdr(e, a, b)
            r['fake02'] += d < 0.02; r['fake10'] += d < 0.10; r['fakeall'] += 1
    r['ntc'] += len(tcs)
    for s, core, base in e['sims']:
        for tag, ok in (('02', core), ('10', True)):
            if ok:
                r['den' + tag] += 1
                r['eff' + tag] += s in found
                r['dupsim' + tag] += nmatch[s] > 1
    return r


SCEN = [
    ('baseline replay', dict()),
    ('C: greedy(score)', dict(mode='grd')),
    ('C: nLayers>score (par)', dict(key='nl>score')),
    ('C: nLayers>score (greedy)', dict(key='nl>score', mode='grd')),
    ('E: unstale', dict(unstale=True)),
    ('D: demote score>50', dict(demote=50.)),
    ('D: demote score>100', dict(demote=100.)),
    ('D: demote score>200', dict(demote=200.)),
    ('D50 + E', dict(demote=50., unstale=True)),
    ('D50 + E + C nl>score(par)', dict(demote=50., unstale=True, key='nl>score')),
    ('D50 + E + C nl>score(grd)', dict(demote=50., unstale=True, key='nl>score', mode='grd')),
    ('D100 + E + C nl>score(par)', dict(demote=100., unstale=True, key='nl>score')),
    ('ORACLE dedup (par)', dict(key='oracle')),
    ('ORACLE dedup + D50 + E', dict(key='oracle', demote=50., unstale=True)),
]


def main():
    if os.path.exists(PK):
        E = pickle.load(open(PK, 'rb'))
    else:
        E = precompute()
        pickle.dump(E, open(PK, 'wb'))
    for hn, sel in (('train(0-49)', lambda e: e['ie'] < 50), ('test(50-99)', lambda e: e['ie'] >= 50),
                    ('all(0-99)', lambda e: True)):
        EE = [e for e in E if sel(e)]
        base = collections.Counter()
        for e in EE:
            base += score_tcs(e, scenario(e))
        # truth baseline for reference
        truth02 = sum(b for e in EE for s, c, b in e['sims'] if c)
        truth10 = sum(b for e in EE for s, c, b in e['sims'])
        print(f"== {hn}: core den {base['den02']} (ntuple eff {truth02 / base['den02']:.4f}, replay eff "
              f"{base['eff02'] / base['den02']:.4f}); dR<0.1 den {base['den10']} (ntuple {truth10 / base['den10']:.4f}, "
              f"replay {base['eff10'] / base['den10']:.4f}); TCs {base['ntc']}, fake TCs {base['fakeall']}")
        for name, kw in SCEN:
            r = collections.Counter()
            for e in EE:
                r += score_tcs(e, scenario(e, **kw))
            d02 = r['eff02'] - base['eff02']; d10 = r['eff10'] - base['eff10']
            print(f"  {name:<30} core {d02:>+4} ({100 * d02 / base['den02']:+.2f}pp) dFake(jdR<.02) "
                  f"{r['fake02'] - base['fake02']:>+4} | dR<0.1 {d10:>+4} ({100 * d10 / base['den10']:+.2f}pp) dFake(<.1) "
                  f"{r['fake10'] - base['fake10']:>+4} | dFakeAll {r['fakeall'] - base['fakeall']:>+5} dTC {r['ntc'] - base['ntc']:>+5}"
                  f" | dDupSim core {r['dupsim02'] - base['dupsim02']:>+3} dR<.1 {r['dupsim10'] - base['dupsim10']:>+3} | demoted {r['ndemoted']}")


if __name__ == '__main__':
    main()
