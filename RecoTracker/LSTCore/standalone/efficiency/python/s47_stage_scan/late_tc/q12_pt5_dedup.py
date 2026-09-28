#!/usr/bin/env python3
"""Q1 + Q2: pT5 dedup (RemoveDupPixelQuintupletsFromMap, Kernels.h:735-768) on the current code.

Q1: attribute the core tracks whose genuine pT5s are all dedup-killed (fake killer / chain-kill / other-sim / tie ...).
Q2: offline evaluation of alternative ranking keys, parallel (as the kernel: any better neighbour kills) and greedy
    (killed only by a KEPT better neighbour), train = events 0-49, test = events 50-99.

Success model (first order): a denominator sim is found if a kept pT5 has frac > 0.75 for it, or a baseline non-pT5 TC
matches it (frac > 0.75) and that TC is a T5 not newly cross-cleaned (CrossCleanT5, TrackCandidate.h:207-276:
>= 4 shared OT hits, |deta|,|dphi| < 0.15) by a pT5 that becomes newly kept.  Ignored 2nd-order effects: pLS of a
newly-dead pT5 becoming a pLS TC; T4 cross-cleaning; T5 of a newly dead pT5 stays partOfPT5 (stale, so no T5 TC).
"""
import sys, collections, json, math
import numpy as np
from common import *

EVAL_DR = (0.02, 0.10)


def fit_circle(x, y):
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc = sol[0], sol[1]
    r = math.sqrt(max(sol[2] + xc * xc + yc * yc, 1e-12))
    return xc, yc, r


def pix_resid(ev, H, p, t5):
    """RMS distance (cm) of the 4 pLS hits to the circle fitted through the T5's 5 MD anchors (x,y)."""
    mds = H.t5_mds(t5)
    x = np.array([H.ax[m] for m in mds], float)
    y = np.array([H.ay[m] for m in mds], float)
    xc, yc, r = fit_circle(x, y)
    px = np.array([ev[f'pLS_hit{k}_x'][p] for k in range(4)], float)
    py = np.array([ev[f'pLS_hit{k}_y'][p] for k in range(4)], float)
    d = np.hypot(px - xc, py - yc) - r
    return float(np.sqrt(np.mean(d * d)))


def dead_parallel(key, nbr):
    n = len(key)
    order_rank = np.lexsort((np.arange(n), key))
    rank = np.empty(n, int)
    rank[order_rank] = np.arange(n)
    return np.array([any(rank[j] < rank[i] for j in nbr[i]) for i in range(n)], bool)


def dead_greedy(key, nbr):
    n = len(key)
    order = np.lexsort((np.arange(n), key))
    kept = np.zeros(n, bool)
    for i in order:
        if not any(kept[j] for j in nbr[i]):
            kept[i] = True
    return ~kept


def main():
    rows = []
    q1 = collections.Counter()
    q1_detail = []
    per_event = []  # per-event data needed for key evaluation
    for ie, ev in events():
        H = Hits(ev)
        n = len(ev['pT5_score'])
        hits, nbr, nmat = replay_pt5_dedup(ev, H)
        sc = ev['pT5_score'].astype(np.float32)
        dup = ev['pT5_isDupReco'].astype(bool)
        fake = ev['pT5_isFake'].astype(bool)
        p_pls = ev['pT5_plsIdx'].astype(int)
        p_t5 = ev['pT5_t5Idx'].astype(int)
        dnn = ev['t5_dnnScore'][p_t5].astype(float) if n else np.zeros(0)
        nl = ev['t5_nLayers'][p_t5].astype(int) if n else np.zeros(0, int)
        quad = ev['pLS_isQuad'][p_pls].astype(bool) if n else np.zeros(0, bool)
        res = np.array([pix_resid(ev, H, p_pls[i], p_t5[i]) for i in range(n)])
        t5fs = np.maximum(ev['t5_t3_fakeScore1'], ev['t5_t3_fakeScore2'])[p_t5] if n else np.zeros(0)
        # genuine sims per pT5 (frac > 0.75 == TC-match criterion) and genuine flag
        gsim = [set(int(s) for s, f in zip(ev['pT5_simIdxAll'][i], ev['pT5_simIdxAllFrac'][i]) if f > TCMATCH)
                for i in range(n)]
        gen = np.array([len(g) > 0 for g in gsim], bool)
        # jet dR of each pT5 (pLS direction) to nearest selected genjet
        # T5 OT hit sets for CrossCleanT5 re-check
        tc_type = ev['tc_type'].astype(int)
        tcsa, tcsf = ev['tc_simIdxAll'], ev['tc_simIdxAllFrac']
        den02 = denominator(ev, 0.02)
        den10 = denominator(ev, 0.10)
        simtc = ev['sim_tcIdx']
        # sims -> list of baseline non-pT5 TCs matched (frac>0.75)
        nonpt5 = collections.defaultdict(list)
        for k in range(len(tc_type)):
            if tc_type[k] == 7:
                continue
            for s, f in zip(tcsa[k], tcsf[k]):
                if f > TCMATCH:
                    nonpt5[int(s)].append(k)
        sims = np.nonzero(den10)[0]
        simdata = []
        for s in sims:
            gp = [i for i in range(n) if s in gsim[i]]
            simdata.append(dict(s=int(s), core=bool(den02[s]), base=bool(simtc[s] >= 0), gp=gp,
                                nonpt5=[(int(k), int(tc_type[k]), int(ev['tc_t5Idx'][k])) for k in nonpt5.get(int(s), [])],
                                dr=float(ev['sim_genjet_deltaR'][s]), pt=float(ev['sim_pt'][s])))
        # T5 OT hits and eta/phi for TC T5s (for CrossCleanT5 re-check)
        t5tc = {}
        for sd in simdata:
            for (k, ty, t5i) in sd['nonpt5']:
                if ty == 4 and t5i >= 0 and t5i not in t5tc:
                    t5tc[t5i] = (set(H.t5_hits(t5i)), float(ev['t5_eta'][t5i]), float(ev['t5_phi'][t5i]))
        pt5_ot = [set(h[4:]) for h in hits]
        pt5_eta = f16(ev['t5_eta'][p_t5]) if n else np.zeros(0)
        pt5_phi = f16(ev['t5_phi'][p_t5]) if n else np.zeros(0)
        # T5-TC killers: which pT5 would cross-clean each TC T5 (>=4 shared OT hits in 0.15 window)
        t5killers = {}
        for t5i, (hs, e, ph) in t5tc.items():
            t5killers[t5i] = [i for i in range(n) if abs(e - pt5_eta[i]) < 0.15 and abs(dphi(ph, pt5_phi[i])) < 0.15
                              and len(hs & pt5_ot[i]) >= 4]
        # jet-dR of fake pT5s: nearest selected genjet
        sel_j = [(float(e), float(p)) for e, p, pt in zip(ev['genjet_eta'], ev['genjet_phi'], ev['genjet_pt'])
                 if pt > 1000 and abs(e) < 2.5]
        pe, pp = ev['pT5_eta'].astype(float), ev['pT5_phi'].astype(float)
        jdr = np.array([min([math.hypot(pe[i] - e, dphi(pp[i], p)) for e, p in sel_j] or [9.]) for i in range(n)])
        per_event.append(dict(ie=ie, n=n, nbr=nbr, sc=sc, dup=dup, fake=fake, gen=gen, dnn=dnn, nl=nl, quad=quad,
                              res=res, t5fs=t5fs, pls=p_pls, t5=p_t5, ptt5=(ev['t5_pt'][p_t5] if n else np.zeros(0)), ptpls=(ev['pLS_pt'][p_pls] if n else np.zeros(0)), rin=(ev['t5_innerRadius'][p_t5] if n else np.zeros(0)), rout=(ev['t5_outerRadius'][p_t5] if n else np.zeros(0)), simdata=simdata, t5killers=t5killers, jdr=jdr,
                              gsim=[sorted(g) for g in gsim]))

        # ---------------- Q1 attribution (core failures with all genuine pT5 dead)
        for sd in simdata:
            if not sd['core'] or sd['base']:
                continue
            gp_all = [int(x) for x in genuine(ev, 'pt5', sd['s'])]  # frac >= 0.75, funnel convention
            if not gp_all or not all(dup[gp_all]):
                continue
            q1['all_killed'] += 1
            cats = set()
            det = []
            for i in gp_all:
                kl = [j for j in nbr[i] if (sc[i] > sc[j]) or (sc[i] == sc[j] and i > j)]
                for j in kl:
                    samep = p_pls[i] == p_pls[j]
                    if sc[i] == sc[j]:
                        c = 'tie'
                    elif dup[j]:
                        c = 'chain(killer dead)'
                    elif fake[j]:
                        c = 'fake_killer_alive'
                    elif sd['s'] in gsim[j]:
                        c = 'same_sim_alive'
                    else:
                        c = 'other_sim_alive'
                    cats.add(c)
                    det.append(dict(i=i, j=j, cat=c, same_pls=bool(samep), same_t5=bool(p_t5[i] == p_t5[j]), sci=float(sc[i]), scj=float(sc[j]),
                                    dnni=float(dnn[i]), dnnj=float(dnn[j]), resi=res[i], resj=res[j],
                                    nli=int(nl[i]), nlj=int(nl[j]), nm=nmat[(i, j)]))
            # per-track category priority: any alive killer that is fake > other-sim > same-sim > chain only > tie
            for c in ['fake_killer_alive', 'other_sim_alive', 'same_sim_alive', 'chain(killer dead)', 'tie']:
                if c in cats:
                    q1['cat:' + c] += 1
                    break
            alive_k = [d for d in det if d['cat'] in ('fake_killer_alive', 'other_sim_alive', 'same_sim_alive')]
            if alive_k:
                q1['alive_killer_same_pls' if any(d['same_pls'] for d in alive_k) else 'alive_killer_diff_pls'] += 1
            if all(d['cat'] == 'chain(killer dead)' for d in det):
                q1['only_chain'] += 1
            q1_detail.append(dict(ev=ie, s=sd['s'], pt=sd['pt'], det=det))
        print(ie, dict(q1), file=sys.stderr, flush=True)

    import pickle
    with open(CACHE + '/../q12_perevent.pkl', 'wb') as f:
        pickle.dump(per_event, f)
    with open('q1_detail.json', 'w') as f:
        json.dump(q1_detail, f, indent=0, default=float)
    print('Q1', dict(q1))


if __name__ == '__main__':
    main()
