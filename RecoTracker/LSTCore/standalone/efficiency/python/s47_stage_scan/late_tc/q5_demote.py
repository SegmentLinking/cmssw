#!/usr/bin/env python3
"""Q3c: 'pixel-mismatch demotion' — emit a surviving pT5 as a T5 TC (drop the pLS) when pixel/T5 compatibility is poor.
Motivation: ~70 core failures have a surviving pT5 TC whose OT part (T5) is genuine but whose pLS belongs to another
particle (frac <= 10/14).  Offline model: demoted pT5 TC -> sims matched = its T5's t5_simIdxAll (T5 frac with
extension hits, >0.75); all other TCs as in the ntuple.  Features are device-available: pT5 score (rPhiChi2),
pixel-to-T5-circle residual, |eta_pLS - eta_T5|, dphi, pT ratio, T5 dnnScore, nLayers, pLS isQuad.
Train = events 0-49, test = 50-99.  Fake = TC matched to no sim (frac>0.75)."""
import sys, collections, math, pickle
import numpy as np
from common import *
from q12_pt5_dedup import pix_resid


def collect():
    out = []
    for ie, ev in events():
        H = Hits(ev)
        tc_type = ev['tc_type'].astype(int)
        p_pls, p_t5 = ev['pT5_plsIdx'].astype(int), ev['pT5_t5Idx'].astype(int)
        den02, den10 = denominator(ev, 0.02), denominator(ev, 0.10)
        sel_j = [(float(e), float(p)) for e, p, pt in zip(ev['genjet_eta'], ev['genjet_phi'], ev['genjet_pt'])
                 if pt > 1000 and abs(e) < 2.5]
        tcs = []
        for k in range(len(tc_type)):
            sims = set(int(s) for s, f in zip(ev['tc_simIdxAll'][k], ev['tc_simIdxAllFrac'][k]) if f > TCMATCH)
            d = dict(type=int(tc_type[k]), sims=sims)
            if tc_type[k] == 7:
                i = int(ev['tc_pt5Idx'][k]); p, t = p_pls[i], p_t5[i]
                d['tsims'] = set(int(s) for s, f in zip(ev['t5_simIdxAll'][t], ev['t5_simIdxAllFrac'][t]) if f > TCMATCH)
                d['psims'] = set(int(s) for s, f in zip(ev['pLS_simIdxAll'][p], ev['pLS_simIdxAllFrac'][p]) if f > TCMATCH)
                d['score'] = float(ev['pT5_score'][i])
                d['res'] = pix_resid(ev, H, p, t)
                d['deta'] = abs(float(ev['pLS_eta'][p]) - float(ev['t5_eta'][t]))
                d['dphi'] = abs(float(dphi(ev['pLS_phi'][p], ev['t5_phi'][t])))
                d['ptr'] = float(ev['pLS_pt'][p]) / max(float(ev['t5_pt'][t]), 1e-3)
                d['dnn'] = float(ev['t5_dnnScore'][t])
                d['nl'] = int(ev['t5_nLayers'][t])
                d['quad'] = bool(ev['pLS_isQuad'][p])
                d['plspt'] = float(ev['pLS_pt'][p])
                d['pls'] = int(p)
                d['plsdup'] = int(ev['pLS_isDup'][p])
            d['jdr'] = min([math.hypot(float(ev['tc_eta'][k]) - e, float(dphi(ev['tc_phi'][k], p))) for e, p in sel_j] or [9.])
            tcs.append(d)
        sims = [dict(s=int(s), core=bool(den02[s]), base=bool(ev['sim_tcIdx'][s] >= 0)) for s in np.nonzero(den10)[0]]
        out.append(dict(ie=ie, tcs=tcs, sims=sims))
        print(ie, file=sys.stderr, flush=True)
    return out


def evaluate(E, demote_fn):
    r = collections.Counter()
    for e in E:
        found = collections.Counter()
        for d in e['tcs']:
            dem = d['type'] == 7 and demote_fn(d)
            ss = d['tsims'] if dem else d['sims']
            for s in ss:
                found[s] += 1
            if dem:
                r['ndemoted'] += 1
                if d['quad'] and d['plsdup'] == 0:
                    r['freed_pls_quad_notdup'] += 1
                    r['freed_pls_genuine'] += bool(d['psims'])
                r['dem_pT5match'] += bool(d['sims'])
                r['dem_T5match'] += bool(d['tsims'])
            for tag, lim in (('02', 0.02), ('10', 0.10), ('all', 99)):
                if d['jdr'] < lim:
                    r['fake' + tag] += (len(ss) == 0)
                    r['fake0' + tag] += (len(d['sims']) == 0)
        for sd in e['sims']:
            new = found[sd['s']] > 0
            for tag, ok in (('02', sd['core']), ('10', True)):
                if ok:
                    r['den' + tag] += 1
                    r['gain' + tag] += new and not sd['base']
                    r['loss' + tag] += sd['base'] and not new
    return r


def fmt(name, r):
    n2, n10 = r['gain02'] - r['loss02'], r['gain10'] - r['loss10']
    return (f"{name:<34} demoted {r['ndemoted']:>5} (pT5-matched {r['dem_pT5match']:>4}, T5-matched {r['dem_T5match']:>4}) | "
            f"core +{r['gain02']:>3} -{r['loss02']:>2} net {n2:>+4} ({100 * n2 / r['den02']:+.2f}pp) dFake {r['fake02'] - r['fake002']:>+4} | "
            f"dR<0.1 +{r['gain10']:>3} -{r['loss10']:>2} net {n10:>+4} ({100 * n10 / r['den10']:+.2f}pp) dFake {r['fake10'] - r['fake010']:>+4}"
            f" | dFakeAll {r['fakeall'] - r['fake0all']:>+5} (base fakeTC all {r['fake0all']})"
            f" | freed quad pLS w/o isDup {r['freed_pls_quad_notdup']} (genuine {r['freed_pls_genuine']})")


def main():
    pk = CACHE + '/../q5_tcs_v2.pkl'
    try:
        E = pickle.load(open(pk, 'rb'))
    except Exception:
        E = collect()
        pickle.dump(E, open(pk, 'wb'))
    # feature separation among surviving pT5 TCs: class c = pT5 unmatched but T5 matched
    cls = collections.defaultdict(list)
    for e in E:
        for d in e['tcs']:
            if d['type'] != 7:
                continue
            c = 'pT5ok' if d['sims'] else ('T5only' if d['tsims'] else 'none')
            if d['sims'] and not (d['sims'] & d['tsims']):
                c = 'pT5ok_T5bad'
            cls[c].append(d)
    print('surviving pT5 TCs by class:', {k: len(v) for k, v in cls.items()})
    for f in ['score', 'res', 'deta', 'dphi', 'ptr', 'dnn', 'nl', 'plspt']:
        line = f'  {f:<6}'
        for c in ['pT5ok', 'pT5ok_T5bad', 'T5only', 'none']:
            v = np.array([d[f] for d in cls[c]], float)
            if len(v):
                q = np.percentile(v, [10, 50, 90])
                line += f' | {c}: {q[0]:.3g} {q[1]:.3g} {q[2]:.3g}'
        print(line)
    print('  quad frac:', {c: round(float(np.mean([d['quad'] for d in v])), 3) for c, v in cls.items()})
    rules = {'baseline (none)': lambda d: False,
             'ALL pT5 -> T5 (unphysical ref)': lambda d: True}
    for x in [0.02, 0.05, 0.1, 0.2]:
        rules[f'res > {x}'] = (lambda x: lambda d: d['res'] > x)(x)
    for x in [0.005, 0.01, 0.02, 0.05]:
        rules[f'deta > {x}'] = (lambda x: lambda d: d['deta'] > x)(x)
    for x in [0.005, 0.01, 0.02]:
        rules[f'dphi > {x}'] = (lambda x: lambda d: d['dphi'] > x)(x)
    for x in [5, 10, 20, 50, 100, 200, 500, 1000]:
        rules[f'score > {x}'] = (lambda x: lambda d: d['score'] > x)(x)
    for x in [1.5, 2.0, 3.0]:
        rules[f'ptratio off by > x{x}'] = (lambda x: lambda d: d['ptr'] > x or d['ptr'] < 1 / x)(x)
    rules['OR(res>0.05, deta>0.02)'] = lambda d: d['res'] > 0.05 or d['deta'] > 0.02
    rules['score>50 & pLSpt>5'] = lambda d: d['score'] > 50 and d['plspt'] > 5
    rules['score>200 OR ptratio x1.5'] = lambda d: d['score'] > 200 or d['ptr'] > 1.5 or d['ptr'] < 1 / 1.5
    rules['ORACLE (T5 matched & pT5 not)'] = lambda d: (not d['sims']) and bool(d['tsims'])
    for hn, sel in (('train(0-49)', lambda e: e['ie'] < 50), ('test(50-99)', lambda e: e['ie'] >= 50),
                    ('all', lambda e: True)):
        EE = [e for e in E if sel(e)]
        print(f'== {hn}')
        for n, fn in rules.items():
            print(fmt(n, evaluate(EE, fn)))


if __name__ == '__main__':
    main()
