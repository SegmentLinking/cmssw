#!/usr/bin/env python3
"""Late-stage attribution for jet-core tracks whose genuine pT5 was built but no TC matched.

READ-ONLY analysis of an existing LST ntuple (--allobj). Exact replay of
RemoveDupPixelQuintupletsFromMap (Kernels.h) using per-pT5 hit sets reconstructed
from MD coordinates (pLS 4 hits + T5 10 hits = 14, as in addPixelQuintupletToMemory),
T5 eta/phi (what pixelQuintuplets.eta/phi hold), pT5_score and GPU index order.

Counterfactual dedup rules evaluated (all keep a SUPERSET of baseline survivors):
  R1 greedy   : sequential NMS by (score, idx): an object is killed only by a KEPT object.
  R2 pixshare : parallel (baseline) but a pair is a duplicate only if the two pT5s share
                >=1 pixel hit (or use the same pLS).
  R3 greedy+pixshare.
"""
import sys, math, collections, json
import numpy as np
import awkward as ak
import uproot

FN = sys.argv[1] if len(sys.argv) > 1 else \
    '/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/Ntuple-files/LSTNtuple_realpls_mastercuts_100evt_v2.root'
NEV = int(sys.argv[2]) if len(sys.argv) > 2 else 100
GEN = 0.75
DETA, DPHI, NM = 0.2, 0.2, 7
CORE = 0.02

BR = ['sim_q', 'sim_pt', 'sim_eta', 'sim_phi', 'sim_vx', 'sim_vy', 'sim_vz', 'sim_genjet_idx', 'sim_genjet_deltaR',
      'genjet_pt', 'genjet_eta', 'sim_tcIdx', 'sim_tcIdxAll', 'sim_tcIdxAllFrac',
      'sim_pt5IdxAll', 'sim_pt5IdxAllFrac', 'sim_pt3IdxAll', 'sim_pt3IdxAllFrac',
      'sim_plsIdxAll', 'sim_plsIdxAllFrac', 'sim_t5IdxAll', 'sim_t5IdxAllFrac',
      'pT5_plsIdx', 'pT5_t5Idx', 'pT5_score', 'pT5_isDupReco', 'pT5_isFake', 'pT5_simIdx',
      'pT5_simIdxAll', 'pT5_simIdxAllFrac',
      'pT3_plsIdx', 'pT3_t3Idx', 'pT3_score', 'pT3_pix_eta', 'pT3_pix_phi', 'pT3_simIdx', 'pT3_isFake',
      'pLS_lsIdx', 'pLS_isQuad', 'pLS_eta', 'pLS_phi', 'pLS_simIdx',
      'ls_mdIdx0', 'ls_mdIdx1', 'md_anchor_x', 'md_anchor_y', 'md_anchor_z', 'md_other_x', 'md_other_y', 'md_other_z',
      't3_lsIdx0', 't3_lsIdx1',
      't5_t3Idx0', 't5_t3Idx1', 't5_eta', 't5_phi', 't5_isDupBits', 't5_partOfPT5', 't5_tightCutFlag',
      't5_triedInPT5',
      'tc_type', 'tc_pt5Idx', 'tc_pt3Idx', 'tc_plsIdx', 'tc_t5Idx', 'tc_nhits', 'tc_isFake', 'tc_simIdx',
      'tc_simIdxAll', 'tc_simIdxAllFrac']


def dphi(a, b):
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


def main():
    t = uproot.open(FN)['tree']
    tot = collections.Counter()
    rows = []          # per core-failing killed-pT5 track
    cf = {r: collections.Counter() for r in ['R1', 'R2', 'R3']}
    val = collections.Counter()
    for chunk_start in range(0, NEV, 10):
        A = t.arrays(BR, entry_start=chunk_start, entry_stop=min(NEV, chunk_start + 10), library='ak')
        for ie in range(len(A)):
            ev = A[ie]
            process_event(ev, chunk_start + ie, tot, rows, cf, val)
        print(f'.. done {min(NEV, chunk_start + 10)} events', file=sys.stderr, flush=True)
    report(tot, rows, cf, val)


def hkey(x, y, z):
    return (round(float(x), 4), round(float(y), 4), round(float(z), 4))


def process_event(ev, ie, tot, rows, cf, val):
    npt5 = len(ev['pT5_score'])
    mdax, mday, mdaz = [np.asarray(ev[k]) for k in ('md_anchor_x', 'md_anchor_y', 'md_anchor_z')]
    mdox, mdoy, mdoz = [np.asarray(ev[k]) for k in ('md_other_x', 'md_other_y', 'md_other_z')]
    ls0, ls1 = np.asarray(ev['ls_mdIdx0']), np.asarray(ev['ls_mdIdx1'])
    t3l0, t3l1 = np.asarray(ev['t3_lsIdx0']), np.asarray(ev['t3_lsIdx1'])
    t5a, t5b = np.asarray(ev['t5_t3Idx0']), np.asarray(ev['t5_t3Idx1'])
    t5eta, t5phi = np.asarray(ev['t5_eta']), np.asarray(ev['t5_phi'])
    t5dup, t5p5 = np.asarray(ev['t5_isDupBits']), np.asarray(ev['t5_partOfPT5'])
    t5tight = np.asarray(ev['t5_tightCutFlag'])
    plsls = np.asarray(ev['pLS_lsIdx'])
    plsquad = np.asarray(ev['pLS_isQuad']).astype(bool)
    plseta, plsphi = np.asarray(ev['pLS_eta']), np.asarray(ev['pLS_phi'])

    def md_hits(m):
        return [hkey(mdax[m], mday[m], mdaz[m]), hkey(mdox[m], mdoy[m], mdoz[m])]

    def ls_mds(l):
        return [int(ls0[l]), int(ls1[l])]

    def pls_hits(p):
        l = int(plsls[p])
        h = []
        for m in ls_mds(l):
            h += md_hits(m)
        return h

    def t3_mds(t3):
        s = []
        for l in (t3l0[t3], t3l1[t3]):
            for m in ls_mds(int(l)):
                if m not in s:
                    s.append(m)
        return s

    def t5_hits(t5):
        mds = []
        for t3 in (t5a[t5], t5b[t5]):
            for m in t3_mds(int(t3)):
                if m not in mds:
                    mds.append(m)
        h = []
        for m in mds:
            h += md_hits(m)
        return h

    p_pls = np.asarray(ev['pT5_plsIdx']).astype(int)
    p_t5 = np.asarray(ev['pT5_t5Idx']).astype(int)
    p_score = np.asarray(ev['pT5_score']).astype(np.float32)
    p_dup = np.asarray(ev['pT5_isDupReco']).astype(bool)
    p_fake = np.asarray(ev['pT5_isFake']).astype(bool)
    p_sim = np.asarray(ev['pT5_simIdx']).astype(int)
    p_simall = ev['pT5_simIdxAll']
    p_simallf = ev['pT5_simIdxAllFrac']
    p_eta = t5eta[p_t5] if npt5 else np.zeros(0)
    p_phi = t5phi[p_t5] if npt5 else np.zeros(0)
    pix = [set(pls_hits(p_pls[i])) for i in range(npt5)]
    ot = [set(t5_hits(p_t5[i])) for i in range(npt5)]
    allh = [list(pls_hits(p_pls[i])) + list(t5_hits(p_t5[i])) for i in range(npt5)]
    allset = [set(h) for h in allh]

    # pairwise overlap graph in window, nMatched counted like checkHitspT5 (count of hits1[i] present in hits2)
    nbr = [[] for _ in range(npt5)]  # (j, pixshare)
    for i in range(npt5):
        for j in range(i + 1, npt5):
            if abs(p_eta[i] - p_eta[j]) > DETA:
                continue
            if abs(dphi(p_phi[i], p_phi[j])) > DPHI:
                continue
            nij = sum(1 for h in allh[i] if h in allset[j])
            nji = sum(1 for h in allh[j] if h in allset[i])
            ps = (p_pls[i] == p_pls[j]) or len(pix[i] & pix[j]) > 0
            if nij >= NM:
                nbr[i].append((j, ps, 'i_by_j'))
            if nji >= NM:
                nbr[j].append((i, ps, 'j_by_i'))

    def beats(j, i):  # j beats i (i removed)
        return (p_score[i] > p_score[j]) or (p_score[i] == p_score[j] and i > j)

    killers = [[j for (j, ps, _) in nbr[i] if beats(j, i)] for i in range(npt5)]
    killers_px = [[j for (j, ps, _) in nbr[i] if beats(j, i) and ps] for i in range(npt5)]
    rep = np.array([len(k) > 0 for k in killers], dtype=bool)
    val['pt5'] += npt5
    val['agree'] += int((rep == p_dup).sum())
    val['rep_dup_true_alive'] += int((rep & ~p_dup).sum())
    val['rep_alive_true_dup'] += int((~rep & p_dup).sum())

    order = sorted(range(npt5), key=lambda i: (float(p_score[i]), i))

    def greedy(kl):
        kept = np.zeros(npt5, bool)
        for i in order:
            if not any(kept[j] for j in kl[i]):
                kept[i] = True
        return kept

    keep = {'R0': ~p_dup, 'R1': greedy(killers),
            'R2': np.array([len(k) == 0 for k in killers_px], bool),
            'R3': greedy(killers_px)}

    # ---------------- sims
    nsim = len(ev['sim_pt'])
    gidx = np.asarray(ev['sim_genjet_idx'])
    gpt, geta = np.asarray(ev['genjet_pt']), np.asarray(ev['genjet_eta'])
    sim_tc = np.asarray(ev['sim_tcIdx'])

    def in_den(i):
        if ev['sim_q'][i] == 0 or not (ev['sim_pt'][i] > 0.9 and abs(ev['sim_eta'][i]) < 4.5):
            return False
        if not (abs(ev['sim_vz'][i]) < 30 and math.hypot(ev['sim_vx'][i], ev['sim_vy'][i]) < 2.5):
            return False
        j = gidx[i]
        return j >= 0 and gpt[j] > 1000 and abs(geta[j]) < 2.5

    den = np.array([in_den(i) for i in range(nsim)], bool)
    dR = np.asarray(ev['sim_genjet_deltaR'])

    # sims having baseline TC (any dR), for duplicate accounting
    tc_type = np.asarray(ev['tc_type'])
    tc_pt5 = np.asarray(ev['tc_pt5Idx'])
    tc_pt3 = np.asarray(ev['tc_pt3Idx'])
    tc_pls = np.asarray(ev['tc_plsIdx'])
    tc_t5 = np.asarray(ev['tc_t5Idx'])
    tc_nh = np.asarray(ev['tc_nhits'])
    pt5_to_tc = {int(tc_pt5[k]): k for k in range(len(tc_type)) if tc_type[k] == 7}
    pls_to_tc = {int(tc_pls[k]): k for k in range(len(tc_type)) if tc_type[k] == 8}
    pt3_to_tc = {int(tc_pt3[k]): k for k in range(len(tc_type)) if tc_type[k] == 5}
    t5_to_tc = {int(tc_t5[k]): k for k in range(len(tc_type)) if tc_type[k] == 4}

    def gen_list(br_i, br_f, i, thr=GEN):
        return [int(x) for x, f in zip(ev[br_i][i], ev[br_f][i]) if f >= thr]

    # pT5 -> list of sims matched > 0.75 (strict, as TC matching)
    p_gsims = [[int(s) for s, f in zip(p_simall[k], p_simallf[k]) if f > GEN] for k in range(npt5)]

    # ---------- counterfactual efficiency & cost
    base_eff = set(i for i in range(nsim) if sim_tc[i] >= 0)
    for R in ('R1', 'R2', 'R3'):
        added = np.where(keep[R] & ~keep['R0'])[0]
        cf[R]['added_pt5'] += len(added)
        cf[R]['added_fake'] += int(p_fake[added].sum())
        newly = set()
        dupadds = 0
        seen = set()
        for k in added:
            ss = p_gsims[k]
            if not ss:
                continue
            s = ss[0]
            if s in base_eff or s in seen:
                dupadds += 1
            else:
                newly.add(s)
            seen.add(s)
        cf[R]['added_dup'] += dupadds
        for s in newly:
            if s < nsim and den[s]:
                cf[R]['gain_den_all'] += 1
                if dR[s] < 0.02:
                    cf[R]['gain_core002'] += 1
                if dR[s] < 0.05:
                    cf[R]['gain_core005'] += 1
                if dR[s] < 0.10:
                    cf[R]['gain_core010'] += 1
        # pLS TCs removed by CrossCleanpLS against added pT5s (shared pixel hit) -> potential loss
        lost = 0
        for plsk, tck in pls_to_tc.items():
            ph = set(pls_hits(plsk))
            if any(len(ph & pix[k]) > 0 for k in added):
                s = int(ev['tc_simIdx'][tck])
                if s >= 0 and s < nsim and den[s]:
                    # does the sim still have another TC?
                    others = [x for x, f in zip(ev['sim_tcIdxAll'][s], ev['sim_tcIdxAllFrac'][s]) if f > GEN and x != tck]
                    newp = [k for k in added if s in p_gsims[k]]
                    if not others and not newp:
                        lost += 1
                        if dR[s] < 0.02:
                            cf[R]['loss_core002_plsTC'] += 1
        cf[R]['loss_den_plsTC'] += lost

    # ------------- per-track attribution for core failures with genuine pT5
    for i in range(nsim):
        if not den[i] or dR[i] >= CORE:
            continue
        tot['den'] += 1
        if sim_tc[i] >= 0:
            continue
        tot['fail'] += 1
        gp5 = gen_list('sim_pt5IdxAll', 'sim_pt5IdxAllFrac', i)
        if not gp5:
            continue
        tot['killed_pool'] += 1
        r = dict(ev=ie, sim=i, pt=float(ev['sim_pt'][i]), dR=float(dR[i]), n_gp5=len(gp5))
        alive = [g for g in gp5 if not p_dup[g]]
        r['n_alive'] = len(alive)
        # genuine at strict > 0.75 (TC matching threshold)
        gp5s = [g for g in gp5 if i in p_gsims[g]]
        r['n_gp5_strict'] = len(gp5s)
        if alive:
            # TC-level: alive pT5 became a TC; why doesn't it match?
            info = []
            for g in alive:
                tck = pt5_to_tc.get(g, -1)
                fr_p = max([f for s, f in zip(p_simall[g], p_simallf[g]) if s == i] or [0])
                if tck < 0:
                    info.append(('noTC', fr_p, -1, -1))
                    continue
                fr_t = max([f for s, f in zip(ev['tc_simIdxAll'][tck], ev['tc_simIdxAllFrac'][tck]) if s == i] or [0])
                info.append(('TC', fr_p, fr_t, int(tc_nh[tck])))
            r['alive_info'] = info
            if any(x[0] == 'noTC' for x in info):
                r['cause'] = 'TC-build: alive pT5 has no TC (overflow?)'
            elif all(x[1] <= GEN for x in info):
                r['cause'] = 'MATCH-THRESHOLD: pT5 frac exactly 0.75 (>=0.75 pool vs >0.75 TC match)'
            elif any(x[1] > GEN and x[2] <= GEN and x[3] > 14 for x in info):
                r['cause'] = 'TC-EXTENSION: ExtendTrackCandidatesFromDupT5 diluted match below 0.75'
            else:
                r['cause'] = 'TC-other'
        else:
            # all genuine pT5 killed by FromMap
            rec = {R: any(keep[R][g] for g in gp5s) for R in ('R1', 'R2', 'R3')}
            r.update({f'rec_{R}': rec[R] for R in rec})
            kinds = collections.Counter()
            pxs = collections.Counter()
            for g in gp5:
                for j in killers[g]:
                    if p_dup[j]:
                        kinds['dead'] += 1
                    elif p_fake[j]:
                        kinds['alive_fake'] += 1
                    elif i in p_gsims[j]:
                        kinds['alive_same'] += 1
                    else:
                        kinds['alive_other'] += 1
                    ps = (p_pls[g] == p_pls[j]) or len(pix[g] & pix[j]) > 0
                    pxs['pix' if ps else 'nopix'] += 1
                    if p_pls[g] == p_pls[j]:
                        pxs['samepls'] += 1
            r['killer_kinds'] = dict(kinds)
            r['killer_pix'] = dict(pxs)
            # the winner-TC does it match another sim which would lose?
            if kinds['alive_fake']:
                r['cause'] = 'pT5-DEDUP: fake winner (CLEAN)'
            elif kinds['alive_other']:
                r['cause'] = 'pT5-DEDUP: other-sim genuine winner (DISPLACEMENT)'
            elif kinds['alive_same']:
                r['cause'] = 'pT5-DEDUP: same-sim <0.75 alive winner'
            else:
                r['cause'] = 'pT5-DEDUP: killed only by DEAD pT5s (CHAIN-KILL)'
        # ------------ fallbacks
        gpls = gen_list('sim_plsIdxAll', 'sim_plsIdxAllFrac', i)
        r['n_gpls'] = len(gpls)
        r['gpls_quad'] = sum(bool(plsquad[p]) for p in gpls)
        r['gpls_inpT5'] = sum(1 for p in gpls if p in set(p_pls.tolist()))
        r['gpls_TC'] = sum(1 for p in gpls if p in pls_to_tc)
        # CrossCleanpLS replay vs pT5/pT3 TCs: shares any pixel hit with a surviving pT5's pLS
        alive_pls = set(p_pls[~p_dup].tolist())
        pt3pls = np.asarray(ev['pT3_plsIdx']).astype(int)
        tc_pt3_pls = set(int(pt3pls[k]) for k in pt3_to_tc)
        cc = []
        for p in gpls:
            if not plsquad[p]:
                cc.append('notQuad')
                continue
            ph = set(pls_hits(p))
            hit_pt5 = any(len(ph & set(pls_hits(q))) > 0 for q in alive_pls)
            hit_pt3 = any(len(ph & set(pls_hits(q))) > 0 for q in tc_pt3_pls)
            if p in alive_pls:
                cc.append('ownPT5alive')
            elif hit_pt5:
                cc.append('CCpLS:sharesPixHit_alivePT5')
            elif hit_pt3:
                cc.append('CCpLS:sharesPixHit_pT3TC')
            else:
                cc.append('other(CheckHitspLS2/T5-embed/dR)')
        r['gpls_fate'] = cc
        gpt3 = gen_list('sim_pt3IdxAll', 'sim_pt3IdxAllFrac', i)
        r['n_gpt3'] = len(gpt3)
        r['gpt3_TC'] = sum(1 for p in gpt3 if p in pt3_to_tc)
        # CrossCleanpT3 replay: pT3 pix eta/phi vs ALL pT5 (alive or dead) pLS eta/phi, dR2<1e-5
        pe, pp = np.asarray(ev['pT3_pix_eta']), np.asarray(ev['pT3_pix_phi'])
        fates = []
        for k in gpt3:
            d2 = (pe[k] - plseta[p_pls]) ** 2 + dphi(pp[k], plsphi[p_pls]) ** 2 if npt5 else np.array([])
            m = d2 < 1e-5
            if m.any():
                fates.append('CCpT3:alive' if (m & ~p_dup).any() else 'CCpT3:DEADpT5only')
            else:
                fates.append('pT3dedup/other')
        r['gpt3_fate'] = fates
        gt5 = gen_list('sim_t5IdxAll', 'sim_t5IdxAllFrac', i)
        r['n_gt5'] = len(gt5)
        f5 = collections.Counter()
        for k in gt5:
            b = int(t5dup[k])
            if b & 1:
                f5['AB'] += 1
            elif t5p5[k]:
                f5['partOfPT5(dead pT5)' if all(p_dup[q] for q in np.where(p_t5 == k)[0]) else 'partOfPT5(alive)'] += 1
            elif b & 0x0E:
                f5['BeforeTC'] += 1
            elif b & 0x10:
                f5['CrossCleanT5'] += 1
            elif not t5tight[k]:
                f5['notTight'] += 1
            elif k in t5_to_tc:
                f5['T5TC(mismatch)'] += 1
            else:
                f5['other'] += 1
        r['gt5_fate'] = dict(f5)
        rows.append(r)


def report(tot, rows, cf, val):
    print('VALIDATION pT5 dedup replay: agree %d/%d  (replay dup but truly alive %d; replay alive but truly dup %d)' % (
        val['agree'], val['pt5'], val['rep_dup_true_alive'], val['rep_alive_true_dup']))
    print('core dR<0.02: denom %d fail %d  genuine-pT5-but-failed pool %d' % (tot['den'], tot['fail'], tot['killed_pool']))
    c = collections.Counter(r['cause'] for r in rows)
    print('\n== ATTRIBUTION (last genuine pT5 removal) ==')
    for k, v in c.most_common():
        print(f'  {v:4d}  {k}')
    killed = [r for r in rows if r['n_alive'] == 0]
    for R in ('R1', 'R2', 'R3'):
        print(f'  recovered by {R}: {sum(r.get("rec_"+R, False) for r in killed)} / {len(killed)} all-killed')
    for cause in sorted(set(r['cause'] for r in killed)):
        sub = [r for r in killed if r['cause'] == cause]
        print(f'   [{cause}] n={len(sub)} R1={sum(r["rec_R1"] for r in sub)} R2={sum(r["rec_R2"] for r in sub)} '
              f'R3={sum(r["rec_R3"] for r in sub)}  killer-pix-sharing: '
              f'{dict(sum((collections.Counter(r["killer_pix"]) for r in sub), collections.Counter()))}')
    print('\n== FALLBACKS for the pool ==')
    print('  genuine pLS present: %d tracks; quad: %d; genuine pLS used in some pT5: %d; pLS became TC (but no match?): %d' % (
        sum(r['n_gpls'] > 0 for r in rows), sum(r['gpls_quad'] > 0 for r in rows),
        sum(r['gpls_inpT5'] > 0 for r in rows), sum(r['gpls_TC'] > 0 for r in rows)))
    fc = collections.Counter()
    for r in rows:
        fs = set(r['gpls_fate'])
        # best (most permissive) fate per track
        for key in ['ownPT5alive', 'other(CheckHitspLS2/T5-embed/dR)', 'CCpLS:sharesPixHit_alivePT5',
                    'CCpLS:sharesPixHit_pT3TC', 'notQuad']:
            if key in fs:
                fc[key] += 1
                break
        else:
            fc['no genuine pLS'] += 1
    print('  per-track pLS-fallback fate (most permissive):', dict(fc))
    print('  genuine pT3 built: %d tracks; became TC: %d' % (sum(r['n_gpt3'] > 0 for r in rows), sum(r['gpt3_TC'] > 0 for r in rows)))
    print('  pT3 fates:', dict(collections.Counter(f for r in rows for f in r['gpt3_fate'])))
    t5c = collections.Counter()
    for r in rows:
        f = r['gt5_fate']
        for key in ['T5TC(mismatch)', 'other', 'notTight', 'CrossCleanT5', 'BeforeTC', 'partOfPT5(alive)',
                    'partOfPT5(dead pT5)', 'AB']:
            if f.get(key):
                t5c[key] += 1
                break
        else:
            t5c['no genuine T5'] += 1
    print('  per-track genuine-T5 fate (most permissive):', dict(t5c))
    print('  genuine-T5 fate object counts:', dict(sum((collections.Counter(r['gt5_fate']) for r in rows), collections.Counter())))
    print('\n== COUNTERFACTUAL DEDUP RULES (all events, all dR) ==')
    for R, d in cf.items():
        print(f'  {R}: {dict(d)}')
    with open(OUT, 'w') as f:
        json.dump(rows, f, indent=1, default=str)
    print('rows ->', OUT)


OUT = '/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/efficiency/python/s44_stage_scan/late_tc/killed_pt5_rows.json'
if __name__ == '__main__':
    if len(sys.argv) > 3:
        OUT = sys.argv[3]
    main()
