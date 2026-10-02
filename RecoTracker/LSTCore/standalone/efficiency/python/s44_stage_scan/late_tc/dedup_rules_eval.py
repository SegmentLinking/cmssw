#!/usr/bin/env python3
"""Offline evaluation of alternative RemoveDupPixelQuintupletsFromMap winner rules + TC-extension effect.

Exact pT5-dedup replay (validated 22208/22209 on real-pLS file) with alternative 'beats' rules:
  R0  baseline parallel, score (rPhiChi2) lower wins, killed even by dead pT5s
  R1  greedy NMS (killed only by KEPT objects), score order
  O   truth oracle parallel: genuine beats fake, else score
  Og  truth oracle greedy
  D   parallel: constituent-T5 dnnScore higher wins (tie -> score)
  DS  parallel: SAME-pLS contests decided by T5 dnnScore, other contests by score
  DSg greedy version of DS (order by score; pair decision DS)
  DM  parallel: score wins unless dnn margin > 0.3 (then dnn wins)
Efficiency proxy per sim: baseline matched non-pT5 TC OR kept pT5 with frac>0.75 (pT5-level, extension ignored).
Also: TC-extension (ExtendTrackCandidatesFromDupT5) net effect on core sims from pT5 vs TC match fractions.
"""
import sys, math, collections
import numpy as np
import uproot

sys.path.insert(0, '/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/efficiency/python/s44_stage_scan/late_tc')
from killed_pt5_attrib import BR, hkey, dphi, GEN, DETA, DPHI, NM

FN = sys.argv[1]
NEV = int(sys.argv[2]) if len(sys.argv) > 2 else 100
BR2 = BR + ['t5_dnnScore', 'tc_pMatched', 't5_score']
RULES = ['R0', 'R1', 'O', 'Og', 'D', 'DS', 'DSg', 'DM', 'K1', 'K2', 'K3', 'K4', 'K5']
BINS = [0.02, 0.05, 0.10, 99]


def run():
    t = uproot.open(FN)['tree']
    eff = {r: collections.Counter() for r in RULES}
    cost = {r: collections.Counter() for r in RULES}
    den = collections.Counter()
    base_eff = collections.Counter()
    ext = collections.Counter()
    val = collections.Counter()
    for s0 in range(0, NEV, 10):
        A = t.arrays(BR2, entry_start=s0, entry_stop=min(NEV, s0 + 10), library='ak')
        for ie in range(len(A)):
            ev_one(A[ie], eff, cost, den, base_eff, ext, val)
        print('..', s0 + 10, file=sys.stderr, flush=True)
    print('replay validation', dict(val))
    print('bins', BINS, ' denom', [den[b] for b in BINS], ' actual TC eff', [base_eff[b] for b in BINS])
    print('%-5s' % 'rule' + ''.join('  eff<%.2f' % b for b in BINS) + '   nKeptPT5  fakePT5  dupPT5')
    for r in RULES:
        print('%-5s' % r + ''.join('  %8d' % eff[r][b] for b in BINS) +
              '   %8d %8d %7d' % (cost[r]['kept'], cost[r]['fake'], cost[r]['dup']))
    print('delta vs R0:')
    for r in RULES[1:]:
        print('%-5s' % r + ''.join('  %+8d' % (eff[r][b] - eff['R0'][b]) for b in BINS) +
              '   %+8d %+8d %+7d' % (cost[r]['kept'] - cost['R0']['kept'], cost[r]['fake'] - cost['R0']['fake'],
                                     cost[r]['dup'] - cost['R0']['dup']))
    print('\nTC-extension effect (alive pT5 -> pT5 TC), sims in denominator:')
    for k in sorted(ext):
        print('  ', k, ext[k])


def ev_one(ev, eff, cost, den, base_eff, ext, val):
    npt5 = len(ev['pT5_score'])
    g = lambda k: np.asarray(ev[k])
    mdax, mday, mdaz, mdox, mdoy, mdoz = [g(k) for k in ('md_anchor_x', 'md_anchor_y', 'md_anchor_z',
                                                        'md_other_x', 'md_other_y', 'md_other_z')]
    ls0, ls1, t3l0, t3l1, t5a, t5b = [g(k) for k in ('ls_mdIdx0', 'ls_mdIdx1', 't3_lsIdx0', 't3_lsIdx1',
                                                    't5_t3Idx0', 't5_t3Idx1')]
    t5eta, t5phi, t5dnn, plsls = g('t5_eta'), g('t5_phi'), g('t5_dnnScore'), g('pLS_lsIdx')

    def mdh(m):
        return [hkey(mdax[m], mday[m], mdaz[m]), hkey(mdox[m], mdoy[m], mdoz[m])]

    def plsh(p):
        l = int(plsls[p]); return mdh(int(ls0[l])) + mdh(int(ls1[l]))

    def t5h(t5):
        mds = []
        for t3 in (t5a[t5], t5b[t5]):
            for l in (t3l0[t3], t3l1[t3]):
                for m in (int(ls0[l]), int(ls1[l])):
                    if m not in mds:
                        mds.append(m)
        h = []
        for m in mds:
            h += mdh(m)
        return h

    pp, pt5 = g('pT5_plsIdx').astype(int), g('pT5_t5Idx').astype(int)
    sc = g('pT5_score').astype(np.float32)
    dup = g('pT5_isDupReco').astype(bool)
    fake = g('pT5_isFake').astype(bool)
    dnn = t5dnn[pt5] if npt5 else np.zeros(0)
    eta = t5eta[pt5] if npt5 else np.zeros(0)
    phi = t5phi[pt5] if npt5 else np.zeros(0)
    allh = [plsh(pp[i]) + t5h(pt5[i]) for i in range(npt5)]
    alls = [set(h) for h in allh]
    gs = [[int(s) for s, f in zip(ev['pT5_simIdxAll'][k], ev['pT5_simIdxAllFrac'][k]) if f > GEN] for k in range(npt5)]
    # directed overlap: ov[i] = list of j such that checkHitspT5(i,j)>=NM (i is the one tested for removal)
    ov = [[] for _ in range(npt5)]
    for i in range(npt5):
        for j in range(i + 1, npt5):
            if abs(eta[i] - eta[j]) > DETA or abs(dphi(phi[i], phi[j])) > DPHI:
                continue
            if sum(1 for h in allh[i] if h in alls[j]) >= NM:
                ov[i].append(j)
            if sum(1 for h in allh[j] if h in alls[i]) >= NM:
                ov[j].append(i)

    def sbeat(j, i):
        return sc[i] > sc[j] or (sc[i] == sc[j] and i > j)

    def dbeat(j, i):
        return dnn[j] > dnn[i] or (dnn[j] == dnn[i] and sbeat(j, i))

    B = {
        'R0': sbeat, 'R1': sbeat,
        'O': lambda j, i: (not fake[j] and fake[i]) or (fake[j] == fake[i] and sbeat(j, i)),
        'D': dbeat,
        'DS': lambda j, i: dbeat(j, i) if pp[i] == pp[j] else sbeat(j, i),
        'DM': lambda j, i: dbeat(j, i) if abs(dnn[j] - dnn[i]) > 0.3 else sbeat(j, i),
    }
    B['Og'] = B['O']
    B['DSg'] = B['DS']

    t5s = g('t5_score')[pt5] if npt5 else np.zeros(0)
    def kbeat(key):
        return lambda j, i: key[i] > key[j] or (key[i] == key[j] and i > j)
    KEYS = {'K1': t5s, 'K2': sc + t5s, 'K3': sc / np.maximum(dnn, 1e-3), 'K4': sc * (1.5 - dnn), 'K5': (sc + t5s) / np.maximum(dnn, 1e-3)}

    def parallel(bf):
        return np.array([not any(bf(j, i) for j in ov[i]) for i in range(npt5)], bool)

    def greedy(bf, order):
        kept = np.zeros(npt5, bool)
        for i in order:
            if not any(kept[j] and bf(j, i) for j in ov[i]):
                kept[i] = True
        return kept

    so = sorted(range(npt5), key=lambda i: (float(sc[i]), i))
    oo = sorted(range(npt5), key=lambda i: (bool(fake[i]), float(sc[i]), i))
    keep = {'R0': parallel(sbeat), 'R1': greedy(sbeat, so), 'O': parallel(B['O']), 'Og': greedy(B['O'], oo),
            'D': parallel(dbeat), 'DS': parallel(B['DS']), 'DSg': greedy(B['DS'], so), 'DM': parallel(B['DM'])}
    for kk, kv in KEYS.items():
        keep[kk] = parallel(kbeat(kv))
    val['agree'] += int((keep['R0'] == ~dup).sum()); val['n'] += npt5

    nsim = len(ev['sim_pt'])
    gidx, gpt, geta = g('sim_genjet_idx'), g('genjet_pt'), g('genjet_eta')
    dR = g('sim_genjet_deltaR')
    tct = g('tc_type')
    tcp = g('tc_pt5Idx')
    # sims with a matched non-pT5 TC
    nonpt5 = set()
    for k in range(len(tct)):
        if tct[k] != 7:
            for s in ev['tc_simIdxAll'][k]:
                nonpt5.add(int(s))
    for r in RULES:
        cost[r]['kept'] += int(keep[r].sum())
        cost[r]['fake'] += int((keep[r] & fake).sum())
        cnt = collections.Counter(gs[k][0] for k in np.where(keep[r])[0] if gs[k])
        cost[r]['dup'] += sum(v - 1 for v in cnt.values())
    simkept = {r: set(s for k in np.where(keep[r])[0] for s in gs[k]) for r in RULES}
    stc = g('sim_tcIdx')
    for i in range(nsim):
        if ev['sim_q'][i] == 0 or not (ev['sim_pt'][i] > 0.9 and abs(ev['sim_eta'][i]) < 4.5):
            continue
        if not (abs(ev['sim_vz'][i]) < 30 and math.hypot(ev['sim_vx'][i], ev['sim_vy'][i]) < 2.5):
            continue
        j = gidx[i]
        if not (j >= 0 and gpt[j] > 1000 and abs(geta[j]) < 2.5):
            continue
        bins = [b for b in BINS if dR[i] < b]
        for b in bins:
            den[b] += 1
            base_eff[b] += int(stc[i] >= 0)
        for r in RULES:
            if i in nonpt5 or i in simkept[r]:
                for b in bins:
                    eff[r][b] += 1
        # extension: alive genuine pT5 (pT5-level >0.75) but sim has no matched TC ; or TC matched but no pT5-level match
        has_p = i in simkept['R0']
        has_tc = stc[i] >= 0
        tag = 'core002' if dR[i] < 0.02 else ('core010' if dR[i] < 0.10 else 'outer')
        if has_p and not has_tc:
            ext[tag + ':LOST_by_extension(pT5>0.75, no TC match)'] += 1
        if has_tc and not has_p and tct[stc[i]] == 7:
            ext[tag + ':GAINED_by_extension(TC pT5-type match, no pT5>0.75)'] += 1


if __name__ == '__main__':
    run()
