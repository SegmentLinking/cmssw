#!/usr/bin/env python3
"""Per-object "rescue threshold" distributions for the oat7 dedup cuts (S44 base, no LST rerun).

Exact offline replay of the three dedup kernels at the S44 commit (7c9dde8e9cc) on an --allobj
ntuple. For every object and every scanned cut (others held at master defaults) it computes the
cut value at which the object would survive:

  PT5_NM  (RemoveDupPixelQuintupletsFromMap): pT5 i dies iff a better-ranked pT5 in the
          |deta|<0.2,|dphi|<0.2 window shares >= NM hits (parallel kernel -> chain kills count).
          t_i = max shared hits with any better-ranked pT5; survives iff NM > t_i.
  AB_NM   (RemoveDupQuintupletsAfterBuild): same idea for T5s on the same lower module,
          window 0.1/0.1, loser = higher score_rphisum (tie -> lower index).
  BTC_*   (RemoveDupQuintupletsBeforeTC): T5s that survived AfterBuild; a pair kills when
          ((dR2 < DR2T or nM >= NM) and d2 < D2L) or (dR2 < DR2L and d2 < D2T).
          For each parameter the critical value is taken over all pairs where the T5 loses;
          pairs that kill through the other branch make it unrescuable by that parameter.

Kills in all three kernels depend only on flags set before the kernel, so single-parameter
survival is exact at that stage. Downstream stages are NOT replayed (see caveats printed).

Usage: [NPROC=40] [SKIP_EVENTS=89] s48_dedup_threshold_hists.py [ntuple] [nevents] [outdir]
(event 89 of LSTNtuple_s44_base_100evt.root has 1.6M T5s; its AfterBuild pair loop takes hours)
"""
import sys, os, math, pickle, collections
import numpy as np
import awkward as ak
import uproot
import multiprocessing as mp
import ctypes

_LIB = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 's48_pairmax.so'))
_P = ctypes.c_void_p
_LIB.pair_max.argtypes = [ctypes.c_int64, _P, _P, ctypes.c_double, _P, _P, ctypes.c_float, ctypes.c_float, _P, _P,
                          ctypes.c_int, ctypes.c_int, _P]


def pair_max(eta, phi, score, H, width, mode, group=None):
    """C++ pair loop: per object, max shared hits over window pairs it loses (see s48_pairmax.cc)."""
    n = len(eta)
    out = np.zeros(n, np.int32)
    if n < 2:
        return out.astype(np.float64)
    key = eta.astype(np.float64) + (group.astype(np.float64) * 100.0 if group is not None else 0.0)
    order = np.ascontiguousarray(np.argsort(key, kind='stable'), dtype=np.int64)
    ks = np.ascontiguousarray(key[order])
    e = np.ascontiguousarray(eta, np.float32)
    ph = np.ascontiguousarray(phi, np.float32)
    sc = np.ascontiguousarray(score, np.float32)
    Hs = np.ascontiguousarray(np.sort(H, axis=1), np.int32)
    _LIB.pair_max(n, order.ctypes.data, ks.ctypes.data, width + 1e-6, e.ctypes.data, ph.ctypes.data, width, width,
                  sc.ctypes.data, Hs.ctypes.data, Hs.shape[1], mode, out.ctypes.data)
    return out.astype(np.float64)

FN = sys.argv[1] if len(sys.argv) > 1 else 'Ntuple-files/LSTNtuple_s44_base_100evt.root'
NEV = int(sys.argv[2]) if len(sys.argv) > 2 else 100
OUT = sys.argv[3] if len(sys.argv) > 3 else 'plots-files/s48_dedup_thresholds'
CORE = 0.02
INF = np.inf

DEF = dict(PT5_NM=7, AB_NM=7, BTC_NM=5, BTC_DR2T=0.001, BTC_D2L=1.0, BTC_DR2L=0.02, BTC_D2T=0.1)
OAT7 = dict(PT5_NM=[9, 11, 13, 14], AB_NM=[8, 9], BTC_NM=[6, 7, 8, 10], BTC_DR2T=[0.0005, 0.0],
            BTC_D2L=[0.5, 0.25], BTC_DR2L=[0.01, 0.005], BTC_D2T=[0.05, 0.025])
# 'int_up': survive iff cut > t (loosen = raise); 'lower': survive iff cut <= v (loosen = lower)
KIND = dict(PT5_NM='int_up', AB_NM='int_up', BTC_NM='int_up',
            BTC_DR2T='lower', BTC_D2L='lower', BTC_DR2L='lower', BTC_D2T='lower')

BR = ['sim_q', 'sim_pt', 'sim_eta', 'sim_vx', 'sim_vy', 'sim_vz', 'sim_genjet_idx', 'sim_genjet_deltaR',
      'genjet_pt', 'genjet_eta', 'genjet_phi', 'sim_tcIdxAll', 'sim_tcIdxAllFrac',
      'md_anchor_x', 'md_anchor_y', 'md_anchor_z', 'md_other_x', 'md_other_y', 'md_other_z', 'md_detId',
      'ls_mdIdx0', 'ls_mdIdx1', 't3_lsIdx0', 't3_lsIdx1', 'pLS_lsIdx',
      't5_t3Idx0', 't5_t3Idx1', 't5_eta', 't5_phi', 't5_score', 't5_embed', 't5_isDupBits',
      't5_partOfPT5', 't5_tightCutFlag', 't5_simIdx',
      'pT5_plsIdx', 'pT5_t5Idx', 'pT5_score', 'pT5_isDupReco', 'pT5_simIdx']


def dphi(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def window_pairs(key, width, group=None):
    """All pairs (a, b), a < b in original index, with |key_a - key_b| <= width (and same group)."""
    n = len(key)
    if n < 2:
        return np.zeros(0, int), np.zeros(0, int)
    k = key.astype(np.float64)
    if group is not None:
        k = k + group.astype(np.float64) * 100.0  # |eta| < 5 so groups never overlap
    order = np.argsort(k, kind='stable')
    ks = k[order]
    hi = np.searchsorted(ks, ks + width, side='right')
    cnt = hi - np.arange(n) - 1
    ii = np.repeat(np.arange(n), cnt)
    start = np.repeat(np.cumsum(cnt) - cnt, cnt)
    jj = ii + 1 + (np.arange(cnt.sum()) - start)
    a, b = order[ii], order[jj]
    return np.minimum(a, b), np.maximum(a, b)


def overlap(H, a, b, chunk=2_000_000):
    out = np.empty(len(a), np.int16)
    for s in range(0, len(a), chunk):
        x, y = H[a[s:s + chunk]], H[b[s:s + chunk]]
        out[s:s + chunk] = (x[:, :, None] == y[:, None, :]).any(2).sum(1)
    return out


def scatter_max(n, idx, val, fill):
    r = np.full(n, fill, dtype=np.float64)
    if len(idx):
        o = np.lexsort((val, idx))
        last = np.r_[idx[o][1:] != idx[o][:-1], True]
        r[idx[o][last]] = np.maximum(r[idx[o][last]], val[o][last])
    return r


def scatter_min(n, idx, val, fill):
    return -scatter_max(n, idx, -np.asarray(val, np.float64), -fill)


def process_event(ev, rec, val):
    # ---- hit ids from MD coordinates (as in S44 killed_pt5_attrib.py)
    ax, ay, az = (np.asarray(ev[k]) for k in ('md_anchor_x', 'md_anchor_y', 'md_anchor_z'))
    ox, oy, oz = (np.asarray(ev[k]) for k in ('md_other_x', 'md_other_y', 'md_other_z'))
    nmd = len(ax)
    xyz = np.round(np.concatenate([np.stack([ax, ay, az], 1), np.stack([ox, oy, oz], 1)]), 4)
    _, hid = np.unique(xyz, axis=0, return_inverse=True)
    hid = hid.reshape(-1)
    mdh = np.stack([hid[:nmd], hid[nmd:]], 1)  # (nmd, 2)
    ls0, ls1 = np.asarray(ev['ls_mdIdx0']), np.asarray(ev['ls_mdIdx1'])
    t3l0, t3l1 = np.asarray(ev['t3_lsIdx0']), np.asarray(ev['t3_lsIdx1'])
    t3md = np.stack([ls0[t3l0], ls1[t3l0], ls1[t3l1]], 1)
    val['t3_md_chain_bad'] += int((ls1[t3l0] != ls0[t3l1]).sum())
    ta, tb = np.asarray(ev['t5_t3Idx0']), np.asarray(ev['t5_t3Idx1'])
    val['t5_md_chain_bad'] += int((t3md[ta, 2] != t3md[tb, 0]).sum())
    t5md = np.concatenate([t3md[ta], t3md[tb][:, 1:]], 1)  # 5 MDs
    H5 = mdh[t5md].reshape(len(ta), 10)
    plsls = np.asarray(ev['pLS_lsIdx'])
    plsH = mdh[np.stack([ls0[plsls], ls1[plsls]], 1)].reshape(len(plsls), 4)

    # ---- sims / jets
    gpt, geta, gphi = (np.asarray(ev[k]) for k in ('genjet_pt', 'genjet_eta', 'genjet_phi'))
    gsel = (gpt > 1000) & (np.abs(geta) < 2.5)
    gidx = np.asarray(ev['sim_genjet_idx'])
    q, spt, seta = np.asarray(ev['sim_q']), np.asarray(ev['sim_pt']), np.asarray(ev['sim_eta'])
    vx, vy, vz = np.asarray(ev['sim_vx']), np.asarray(ev['sim_vy']), np.asarray(ev['sim_vz'])
    gj = np.where(gidx >= 0, gidx, 0)
    den_all = (q != 0) & (spt > 0.9) & (np.abs(seta) < 4.5) & (np.abs(vz) < 30) & (np.hypot(vx, vy) < 2.5)
    den = den_all & (gidx >= 0) & gsel[gj] if len(gpt) else np.zeros(len(q), bool)
    sdr = np.asarray(ev['sim_genjet_deltaR'])
    found = np.zeros(len(q), bool)
    for s, (ids, fr) in enumerate(zip(ev['sim_tcIdxAll'], ev['sim_tcIdxAllFrac'])):
        if len(fr) and max(fr) > 0.75:
            found[s] = True

    def obj_dr(eta, phi):
        if not gsel.any() or len(eta) == 0:
            return np.full(len(eta), 9.0)
        e, p = geta[gsel], gphi[gsel]
        return np.sqrt((eta[:, None] - e[None]) ** 2 + dphi(phi[:, None], p[None]) ** 2).min(1)

    t5eta, t5phi = np.asarray(ev['t5_eta']), np.asarray(ev['t5_phi'])
    t5s = np.asarray(ev['t5_score']).astype(np.float32)
    bits = np.asarray(ev['t5_isDupBits'])
    n5 = len(t5eta)
    lowmod = np.asarray(ev['md_detId'])[t5md[:, 0]]

    # ================= AfterBuild (AB_NM)
    t_ab = pair_max(t5eta, t5phi, t5s, H5, 0.1, 0, group=np.unique(lowmod, return_inverse=True)[1].reshape(-1))
    ab_dead_rep = t_ab >= DEF['AB_NM']
    ab_dead = (bits & 1) != 0
    val['ab_n'] += n5
    val['ab_agree'] += int((ab_dead_rep == ab_dead).sum())

    # ================= BeforeTC (BTC_*), on AB survivors
    el = np.where(~ab_dead)[0]
    p5 = np.asarray(ev['t5_partOfPT5']).astype(bool)
    emb = np.asarray(ak.to_numpy(ev['t5_embed'])) if n5 else np.zeros((0, 6))
    btc = {k: None for k in ('BTC_NM', 'BTC_DR2T', 'BTC_D2L', 'BTC_DR2L', 'BTC_D2T')}
    a, b = window_pairs(t5eta[el], 0.1)
    a, b = el[a], el[b]
    dE = np.abs(t5eta[a] - t5eta[b])
    dP = np.abs(dphi(t5phi[a], t5phi[b]))
    m = (dE <= 0.1) & (dP <= 0.1) & ~(p5[a] & p5[b])
    a, b, dE, dP = a[m], b[m], dE[m], dP[m]
    dE32 = dE.astype(np.float32)
    dP32 = dP.astype(np.float32)
    dR2 = (dE32 * dE32 + dP32 * dP32).astype(np.float64)
    nmb = overlap(H5, a, b).astype(np.float64)
    d2 = ((emb[a].astype(np.float32) - emb[b].astype(np.float32)) ** 2).sum(1).astype(np.float64)
    # loser: from ix=a perspective (symmetric outcome)
    s1, s2 = t5s[a], t5s[b]
    a_loses = p5[b] | (s1 > s2)
    b_loses = ~a_loses & (p5[a] | (s1 < s2))
    L = np.where(a_loses, a, np.where(b_loses, b, a))  # tie -> min index = a
    D = DEF
    brA = lambda nmc, dr2t, d2l: ((dR2 < dr2t) | (nmb >= nmc)) & (d2 < d2l)
    brB = lambda dr2l, d2t: (dR2 < dr2l) & (d2 < d2t)
    kill0 = brA(D['BTC_NM'], D['BTC_DR2T'], D['BTC_D2L']) | brB(D['BTC_DR2L'], D['BTC_D2T'])
    btc_dead_rep = np.zeros(n5, bool)
    btc_dead_rep[L[kill0]] = True
    btc_dead = (bits & 0b1110) != 0
    val['btc_n'] += len(el)
    val['btc_agree'] += int((btc_dead_rep[el] == btc_dead[el]).sum())
    # per-parameter critical values
    # NM: survive iff NM > t ; independent kill -> t = +inf
    indep = (dR2 < D['BTC_DR2T']) & (d2 < D['BTC_D2L']) | brB(D['BTC_DR2L'], D['BTC_D2T'])
    dep = ~indep & (d2 < D['BTC_D2L'])
    t = np.where(indep, INF, np.where(dep, nmb, -INF))
    btc['BTC_NM'] = scatter_max(n5, L, t, 0)
    # lower-type: survive iff cut <= v ; independent -> v = -inf (never) ; no dependence -> +inf
    def lower(indep, dep, crit):
        v = np.where(indep, -INF, np.where(dep, crit, INF))
        return scatter_min(n5, L, v, INF)
    btc['BTC_DR2T'] = lower((nmb >= D['BTC_NM']) & (d2 < D['BTC_D2L']) | brB(D['BTC_DR2L'], D['BTC_D2T']),
                            d2 < D['BTC_D2L'], dR2)
    btc['BTC_D2L'] = lower(brB(D['BTC_DR2L'], D['BTC_D2T']), (dR2 < D['BTC_DR2T']) | (nmb >= D['BTC_NM']), d2)
    btc['BTC_DR2L'] = lower(brA(D['BTC_NM'], D['BTC_DR2T'], D['BTC_D2L']), d2 < D['BTC_D2T'], dR2)
    btc['BTC_D2T'] = lower(brA(D['BTC_NM'], D['BTC_DR2T'], D['BTC_D2L']), dR2 < D['BTC_DR2L'], d2)
    val['btc_nm_selfcheck'] += int(((btc['BTC_NM'][el] >= D['BTC_NM']) == btc_dead_rep[el]).sum())

    # ================= pT5 FromMap (PT5_NM)
    pls, pt5t5 = np.asarray(ev['pT5_plsIdx']).astype(int), np.asarray(ev['pT5_t5Idx']).astype(int)
    np5 = len(pls)
    H14 = np.concatenate([plsH[pls], H5[pt5t5]], 1) if np5 else np.zeros((0, 14), int)
    pe, pp = t5eta[pt5t5], t5phi[pt5t5]
    ps = np.asarray(ev['pT5_score']).astype(np.float32)
    t_pt5 = pair_max(pe, pp, ps, H14, 0.2, 1)
    val['pt5_n'] += np5
    val['pt5_agree'] += int(((t_pt5 >= DEF['PT5_NM']) == np.asarray(ev['pT5_isDupReco']).astype(bool)).sum())

    # ================= records
    t5sim = np.asarray(ev['t5_simIdx'])
    t5dr = obj_dr(t5eta, t5phi)
    t5elig = ~p5 & (np.asarray(ev['t5_tightCutFlag']) != 0)
    ev_id = val['events']
    rec['T5_AB'].append(dict(ev=ev_id, v=t_ab, sim=t5sim, dr=t5dr, elig=np.ones(n5, bool), mask=np.ones(n5, bool)))
    for k, v in btc.items():
        msk = np.zeros(n5, bool)
        msk[el] = True
        rec[k].append(dict(ev=ev_id, v=v, sim=t5sim, dr=t5dr, elig=t5elig, mask=msk))
    rec['PT5_NM'].append(dict(ev=ev_id, v=t_pt5, sim=np.asarray(ev['pT5_simIdx']), dr=obj_dr(pe, pp),
                              elig=np.ones(np5, bool), mask=np.ones(np5, bool)))
    rec['sims'].append(dict(ev=ev_id, den=den, den_all=den_all, found=found, dr=sdr))
    val['events'] += 1


def work(ie):
    t = uproot.open(FN)['tree']
    ev = t.arrays(BR, entry_start=ie, entry_stop=ie + 1, library='ak')[0]
    rec = collections.defaultdict(list)
    val = collections.Counter()
    process_event(ev, rec, val)
    for k in rec:
        for r in rec[k]:
            r['ev'] = ie
    return dict(rec), val


def main():
    os.makedirs(OUT, exist_ok=True)
    rec = collections.defaultdict(list)
    val = collections.Counter()
    with mp.Pool(int(os.environ.get('NPROC', '50'))) as pool:
        skip = {int(x) for x in os.environ.get('SKIP_EVENTS', '').split(',') if x}
        evs = [e for e in range(NEV) if e not in skip]
        for n, (r, v) in enumerate(pool.imap_unordered(work, evs)):
            for k in r:
                rec[k] += r[k]
            val.update(v)
            print(f'.. {n + 1} events', file=sys.stderr, flush=True)
    with open(os.path.join(OUT, 'records.pkl'), 'wb') as f:
        pickle.dump(dict(rec=dict(rec), val=dict(val), fn=FN, nev=len(evs), skipped=sorted(skip)), f)
    print('VALIDATION', dict(val))


if __name__ == '__main__':
    main()
