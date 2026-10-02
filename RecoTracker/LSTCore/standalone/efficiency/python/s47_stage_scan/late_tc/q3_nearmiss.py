#!/usr/bin/env python3
"""Q3b: near-miss TCs of core failing sims.  The ntuple's sim_tcIdxAll only stores frac > 0.75, so the fraction is
recomputed here: TC hits rebuilt from objects (coordinate keys), sim hits = sim_recoHitX/Y/Z; frac = shared unique /
unique TC hits (as matchedSimTrkIdxsAndFracs, trkCore.cc:351+, dedups hits).  sim_recoHit has OT hits only, so the\npixel part counts as matched iff the TC's pLS is genuine for the sim.  Validated on matched sims."""
import sys, collections, json
import numpy as np
import uproot
from common import *

FN = '/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/Ntuple-files/LSTNtuple_s45_rebased_100evt.root'
TN = {7: 'pT5', 4: 'T5', 5: 'pT3', 8: 'pLS', 9: 'T4'}


def main():
    tree = uproot.open(FN)['tree']
    C = collections.Counter()
    val = collections.Counter()
    rows = []
    RH = None
    for ie, ev in events():
        if ie % 10 == 0:
            RH = tree.arrays(['sim_recoHitX', 'sim_recoHitY', 'sim_recoHitZ'], entry_start=ie, entry_stop=ie + 10,
                             library='np')
        rx, ry, rz = RH['sim_recoHitX'][ie % 10], RH['sim_recoHitY'][ie % 10], RH['sim_recoHitZ'][ie % 10]
        H = Hits(ev)
        den = denominator(ev, 0.02)
        tc_type = ev['tc_type'].astype(int)
        ntc = len(tc_type)
        p_pls, p_t5 = ev['pT5_plsIdx'].astype(int), ev['pT5_t5Idx'].astype(int)
        tch = []
        parts = []
        for k in range(ntc):
            ty = tc_type[k]
            if ty == 7:
                i = int(ev['tc_pt5Idx'][k]); pix = H.pls_hits(p_pls[i]); ot = H.t5_hits(p_t5[i])
                parts.append(('pls', int(p_pls[i]), 't5', int(p_t5[i])))
            elif ty == 4:
                pix = []; ot = H.t5_hits(int(ev['tc_t5Idx'][k])); parts.append(('t5', int(ev['tc_t5Idx'][k])))
            elif ty == 5:
                i = int(ev['tc_pt3Idx'][k]); pix = H.pls_hits(int(ev['pT3_plsIdx'][i]))
                ot = H.t3_hits(int(ev['pT3_t3Idx'][i])); parts.append(('pls', int(ev['pT3_plsIdx'][i])))
            elif ty == 9:
                pix = []; ot = H.t4_hits(int(ev['tc_t4Idx'][k])); parts.append(('t4', int(ev['tc_t4Idx'][k])))
            else:
                pix = H.pls_hits(int(ev['tc_plsIdx'][k])); ot = []; parts.append(('pls', int(ev['tc_plsIdx'][k])))
            tch.append((list(dict.fromkeys(pix)), list(dict.fromkeys(ot))))
        teta, tphi = ev['tc_eta'], ev['tc_phi']
        simtc = ev['sim_tcIdx']
        for s in np.nonzero(den)[0]:
            if s >= len(rx):
                continue
            sh = set(('c', round(float(a), 4), round(float(b), 4), round(float(c), 4)) for a, b, c in zip(rx[s], ry[s], rz[s]))
            # also map index-keyed hits: none needed (all T5 hits mapped to coords)
            cand = np.nonzero((np.abs(teta - ev['sim_eta'][s]) < 0.3) & (np.abs(dphi(tphi, ev['sim_phi'][s])) < 0.3))[0]
            best, bk, bd = 0.0, -1, None
            gpls = set(int(x) for x in genuine(ev, 'pls', s))
            for k in cand:
                pix, ot = tch[k]
                if not (pix or ot):
                    continue
                # sim_recoHit* holds OT hits only: pixel part counted as fully matched iff its pLS is genuine (>0.75)
                npx = len(pix) if (parts[k][0] == 'pls' and parts[k][1] in gpls) else 0
                no = sum(h in sh for h in ot)
                f = (npx + no) / (len(pix) + len(ot))
                if f > best:
                    best, bk, bd = f, int(k), (npx, len(pix), no, len(ot))
            if simtc[s] >= 0:
                val['matched'] += 1
                val['recomputed>0.75'] += best > 0.75
                continue
            C['fail'] += 1
            b = 'none' if best == 0 else ('<0.5' if best < 0.5 else ('0.5-0.6' if best < 0.6 else (
                '0.6-0.75' if best <= 0.75 else '>0.75 (?)')))
            C['best ' + b] += 1
            if best >= 0.6:
                ty = TN[int(tc_type[bk])]
                npx, lp, no, lo = bd
                C[f'nearmiss {ty} pix {npx}/{lp} OT {no}/{lo}'] += 1
                g = {o: [int(x) for x in genuine(ev, o, s)] for o in ['pls', 't5']}
                rows.append(dict(ev=ie, s=int(s), best=best, type=ty, pix=f'{npx}/{lp}', ot=f'{no}/{lo}',
                                 parts=parts[bk], has_gen_pls=bool(g['pls']), has_gen_t5=bool(g['t5']),
                                 pt=float(ev['sim_pt'][s])))
        print(ie, dict(val), file=sys.stderr, flush=True)
    print('validation (matched sims whose recomputed best frac > 0.75):', dict(val))
    for k, v in sorted(C.items(), key=lambda x: -x[1]):
        print(f'{v:>6}  {k}')
    json.dump(rows, open('q3_nearmiss_rows.json', 'w'), indent=0)


if __name__ == '__main__':
    main()
