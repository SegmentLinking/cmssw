#!/usr/bin/env python3
"""
S47 early_pixel: shared helpers. Emulates LST's prepareInput() seed selection
(interface/LSTPrepareInput.h:105-139) and CheckHitspLS pass 1 (src/alpaka/Kernels.h:772-863)
from the INPUT trackingNtuple, so the pass-1 decision can be separated from the later
CrossCleanpLS `isDup = true` writes (TrackCandidate.h:334-366) that share bit0 in pLS_isDup.
Read-only.
"""
import collections
import numpy as np

PTCUT = 0.8
LST_ALGOS = (4, 22)


def fp(x):
    return np.asarray(x, dtype=np.float32).tobytes()


def event_map(lst_simpt, in_simpt, in_bx, in_ev):
    """LST event -> input event, by accepted-sim sim_pt fingerprint."""
    inmap = {}
    for e in range(len(in_simpt)):
        acc = (np.asarray(in_bx[e]) == 0) & (np.asarray(in_ev[e]) == 0)
        inmap[fp(np.asarray(in_simpt[e])[acc])] = e
    return {le: inmap[fp(lst_simpt[le])] for le in range(len(lst_simpt)) if fp(lst_simpt[le]) in inmap}


def hit_sims(idx, typ, pix_sh, ph2_sh, sh_trk):
    lst = pix_sh[idx] if typ == 0 else ph2_sh[idx]
    return set(int(sh_trk[k]) for k in np.asarray(lst))


def build_seeds(E):
    """E: dict of per-event numpy/awkward arrays of the input ntuple. Returns list of seed dicts."""
    algo = np.asarray(E["see_algo"])
    pxl = np.asarray(E["see_stateTrajGlbPx"], dtype=np.float64)
    pyl = np.asarray(E["see_stateTrajGlbPy"], dtype=np.float64)
    pzl = np.asarray(E["see_stateTrajGlbPz"], dtype=np.float64)
    ptIn = np.hypot(pxl, pyl)
    etaLH = np.arcsinh(pzl / ptIn)
    ptErr = np.asarray(E["see_ptErr"])
    sh_trk = np.asarray(E["simhit_simTrkIdx"])
    pix_sh, ph2_sh = E["pix_simHitIdx"], E["ph2_simHitIdx"]
    # score_lsq (Segment.h:1140-1145): slope=sinh(eta_PCA); anchors r3PCA (inner) and r3LH (outer)
    px = np.asarray(E["see_px"], dtype=np.float64); py = np.asarray(E["see_py"], dtype=np.float64)
    pz = np.asarray(E["see_pz"], dtype=np.float64)
    dxy = np.asarray(E["see_dxy"], dtype=np.float64); dz = np.asarray(E["see_dz"], dtype=np.float64)
    pt = np.hypot(px, py); p = np.sqrt(pt * pt + pz * pz)
    vz = (dz * pt * pt / p / p).astype(np.float32)
    vx = (-dxy * py / pt - px / p * pz / p * dz).astype(np.float32)
    vy = (dxy * px / pt - py / p * pz / p * dz).astype(np.float32)
    slope = np.sinh(np.arcsinh(pz / pt).astype(np.float32)).astype(np.float32)
    intercept = vz - slope * np.sqrt(vx * vx + vy * vy).astype(np.float32)
    X = np.asarray(E["see_stateTrajGlbX"], dtype=np.float32); Y = np.asarray(E["see_stateTrajGlbY"], dtype=np.float32)
    Z = np.asarray(E["see_stateTrajGlbZ"], dtype=np.float32)
    sc = (np.sqrt(X * X + Y * Y) * slope + intercept) - Z
    score = (sc * sc).astype(np.float32)
    hidx, htyp = E["see_hitIdx"], E["see_hitType"]
    seeds = []
    for s in range(len(algo)):
        hi = [int(x) for x in np.asarray(hidx[s])]
        ht = [int(x) for x in np.asarray(htyp[s])]
        uniq = []
        for a, b in zip(hi, ht):
            if (a, b) not in uniq:
                uniq.append((a, b))
        cnt = collections.Counter()
        for a, b in uniq:
            for t in hit_sims(a, b, pix_sh, ph2_sh, sh_trk):
                cnt[t] += 1
        n = len(uniq)
        fr = {t: c / n for t, c in cnt.items() if t >= 0}
        good_algo = int(algo[s]) in LST_ALGOS
        good_pt = ptIn[s] > PTCUT - 2 * ptErr[s]
        # pLSHitsIdxs order (Segment.h:1147-1150): {h0, h2, h1, h3 or (triplet) h2 repeated}
        keys = [(b, a) for a, b in zip(hi, ht)]
        if len(keys) >= 3:
            last = keys[-1] if len(keys) > 3 else keys[2]
            ph = (keys[0], keys[2], keys[1], last)
        else:
            ph = tuple(keys)
        seeds.append(dict(algo=int(algo[s]), n=n, nraw=len(hi), fr=fr, ptIn=float(ptIn[s]), ptErr=float(ptErr[s]),
                          eta=float(etaLH[s]), phi=float(np.arctan2(pyl[s], pxl[s])), score=float(score[s]),
                          lst=bool(good_algo and good_pt), good_algo=good_algo, good_pt=bool(good_pt), ph=ph,
                          quad=len(hi) > 3))
    return seeds


def pair_list(etas):
    """All (i<j) pairs with |deta| <= 0.1 (Kernels.h:800)."""
    etas = np.asarray(etas)
    order = np.argsort(etas, kind="stable")
    es = etas[order]
    out = []
    for a in range(len(order)):
        hi = np.searchsorted(es, es[a] + 0.1, side="right")
        for b in range(a + 1, hi):
            i, j = order[a], order[b]
            if abs(etas[i] - etas[j]) > 0.1:
                continue
            out.append((min(i, j), max(i, j)))
    return out


def checkhits_pass1(ph, quad, score, pairs, distinct=False):
    """Kernels.h:789-850, secondpass=false. Returns isdup(bool array), killers{victim:[(winner,npm,ndist)]}."""
    n = len(ph)
    isdup = np.zeros(n, bool)
    killers = collections.defaultdict(list)
    for i, j in pairs:
        p1, p2 = ph[i], ph[j]
        if distinct:
            npm = len(set(p1) & set(p2))
        else:
            npm = sum(1 for h in p1 if h in p2)
        if npm < 3:
            continue
        qd = int(quad[i]) - int(quad[j])
        sd = score[i] - score[j]
        if qd > 0: rm = j
        elif qd < 0: rm = i
        elif sd < 0: rm = j
        elif sd > 0: rm = i
        else: rm = i
        win = j if rm == i else i
        isdup[rm] = True
        killers[rm].append((win, npm, len(set(p1) & set(p2))))
    return isdup, killers


def rank_key(quad, score, i):
    # higher priority first: quad, then lower score, then (tie -> larger index kept, Kernels.h:818-819)
    return (-int(quad[i]), score[i], -i)


def checkhits_nms(ph, quad, score, pairs, distinct=False):
    """Variant (c): a pLS is removed only by an UNREMOVED higher-priority partner (greedy NMS in the
    same priority order). Returns isdup."""
    n = len(ph)
    nb = collections.defaultdict(list)
    for i, j in pairs:
        p1, p2 = ph[i], ph[j]
        npm = len(set(p1) & set(p2)) if distinct else sum(1 for h in p1 if h in p2)
        if npm >= 3:
            nb[i].append(j); nb[j].append(i)
    order = sorted(range(n), key=lambda i: rank_key(quad, score, i))
    pos = {i: k for k, i in enumerate(order)}
    isdup = np.zeros(n, bool)
    for i in order:
        for j in nb[i]:
            if pos[j] < pos[i] and not isdup[j]:
                isdup[i] = True
                break
    return isdup
