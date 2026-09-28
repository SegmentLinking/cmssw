#!/usr/bin/env python3
"""Q2: offline evaluation of pT5-dedup ranking keys (lower key wins) on the per-event data of q12_pt5_dedup.py.
Modes: 'par' = kernel semantics (any better neighbour kills, no isDup check -> chain kills),
       'grd' = greedy NMS (killed only by a kept better neighbour).
Split: train = events 0-49, test = events 50-99.  Numbers = gained / lost denominator sims vs the ntuple baseline,
and Δ of kept FAKE pT5s (pT5_isFake) within dR<0.02 / <0.10 of a selected genjet (pt>1 TeV, |eta|<2.5) and overall."""
import sys, pickle, collections
import numpy as np
sys.path.insert(0, '.')
from common import CACHE
from q12_pt5_dedup import dead_parallel, dead_greedy

PE = pickle.load(open(CACHE + '/../q12_perevent.pkl', 'rb'))
L = lambda x: np.log(np.maximum(np.asarray(x, float), 1e-4))

KEYS = {
    'score (master)': lambda e: e['sc'].astype(float),
    '-dnn': lambda e: -e['dnn'],
    'score/dnn^0.5': lambda e: L(e['sc']) - 0.5 * L(e['dnn']),
    'score/dnn': lambda e: L(e['sc']) - 1.0 * L(e['dnn']),
    'score/dnn^2': lambda e: L(e['sc']) - 2.0 * L(e['dnn']),
    'score/dnn^4': lambda e: L(e['sc']) - 4.0 * L(e['dnn']),
    'pixres': lambda e: e['res'],
    'score*pixres^0.5': lambda e: L(e['sc']) + 0.5 * L(e['res']),
    'score*pixres': lambda e: L(e['sc']) + 1.0 * L(e['res']),
    'score*pixres^2': lambda e: L(e['sc']) + 2.0 * L(e['res']),
    'pixres/dnn': lambda e: L(e['res']) - L(e['dnn']),
    'score*pixres/dnn': lambda e: L(e['sc']) + L(e['res']) - L(e['dnn']),
    'nLayers>score': lambda e: -100.0 * e['nl'] + L(e['sc']),
    'nLayers>score*pixres': lambda e: -100.0 * e['nl'] + L(e['sc']) + L(e['res']),
    'ORACLE genuine>score': lambda e: 1000.0 * (~e['gen']) + L(e['sc']),
}


def evaluate(key_fn, mode, evs):
    r = collections.Counter()
    for e in evs:
        n = e['n']
        if n:
            key = key_fn(e)
            dead = dead_parallel(key, e['nbr']) if mode == 'par' else dead_greedy(key, e['nbr'])
        else:
            dead = np.zeros(0, bool)
        kept = ~dead
        base_kept = ~e['dup']
        newly = kept & ~base_kept
        for tag, lim in (('02', 0.02), ('10', 0.10), ('all', 99.)):
            m = e['jdr'] < lim
            r['dfake' + tag] += int((kept & e['fake'] & m).sum()) - int((base_kept & e['fake'] & m).sum())
        r['dpt5'] += int(kept.sum()) - int(base_kept.sum())
        for sd in e['simdata']:
            gp = sd['gp']
            via_pt5 = any(kept[i] for i in gp)
            via_other = False
            for (k, ty, t5i) in sd['nonpt5']:
                if ty == 4 and any(newly[i] for i in e['t5killers'].get(t5i, [])):
                    continue
                via_other = True
            new = via_pt5 or via_other
            ndup_new = sum(1 for i in gp if kept[i])
            ndup_old = sum(1 for i in gp if base_kept[i])
            for tag, ok in (('02', sd['core']), ('10', True)):
                if not ok:
                    continue
                r['den' + tag] += 1
                r['base' + tag] += sd['base']
                r['gain' + tag] += (new and not sd['base'])
                r['loss' + tag] += (sd['base'] and not new)
                r['ddupl' + tag] += (ndup_new > 1) - (ndup_old > 1)
    return r


def row(name, mode, r):
    net02 = r['gain02'] - r['loss02']
    net10 = r['gain10'] - r['loss10']
    return (f"{name:<24}{mode:>4} | core: +{r['gain02']:>3} -{r['loss02']:>3} net {net02:>+4} ({100 * net02 / r['den02']:+.2f}pp)"
            f" dFake {r['dfake02']:>+4} | dR<0.1: +{r['gain10']:>3} -{r['loss10']:>3} net {net10:>+4} "
            f"({100 * net10 / r['den10']:+.2f}pp) dFake {r['dfake10']:>+4} | dFakeAll {r['dfakeall']:>+5} dpT5 {r['dpt5']:>+5}"
            f" dDupSim02 {r['ddupl02']:>+3}")


def main():
    halves = {'train(0-49)': [e for e in PE if e['ie'] < 50], 'test(50-99)': [e for e in PE if e['ie'] >= 50],
              'all(0-99)': PE}
    for hn, evs in halves.items():
        r0 = evaluate(KEYS['score (master)'], 'par', evs)
        print(f"== {hn}: core den {r0['den02']}, base eff {r0['base02'] / r0['den02']:.4f};  "
              f"dR<0.1 den {r0['den10']}, base eff {r0['base10'] / r0['den10']:.4f}")
        # 'keep every genuine pT5' ceiling = oracle with no fake kill constraint
        for name, fn in KEYS.items():
            for mode in ('par', 'grd'):
                print(row(name, mode, evaluate(fn, mode, evs)))
        print()


if __name__ == '__main__':
    main()
