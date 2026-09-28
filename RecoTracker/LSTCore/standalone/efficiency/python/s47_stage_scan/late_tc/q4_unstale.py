#!/usr/bin/env python3
"""Q4: stale partOfPT5.  quintuplets.partOfPT5 is set when a pT5 is built (PixelQuintuplet.h:791) and never cleared
when RemoveDupPixelQuintupletsFromMap kills that pT5 (Kernels.h:762 only sets pixelQuintuplets.isDup).  A T5 with
partOfPT5 is never a TC (TrackCandidate.h:568), is skipped by CrossCleanT5 (:224) and is exempt from BeforeTC vs
another partOfPT5 T5 (Kernels.h:547).

Fix model ('unstale'): partOfPT5' = partOfPT5 AND at least one pT5 using the T5 is alive.  Replays with the new flags:
  RemoveDupQuintupletsBeforeTC (Kernels.h:498-586)  and  CrossCleanT5 (TrackCandidate.h:207-276),
then the T5-TC set, CrossCleanT4 losses (T4 TC sharing >= 3 hits with a new T5 TC, TrackCandidate.h:419-422).
Baseline replays are validated against the ntuple bits / T5 TC set first."""
import sys, collections
import numpy as np
from common import *


def btc_dead(ev, H, part, abdup, t5hits, t5sets, idxs):
    """isDup bit 2 replay: i dies if any j (not AB-dup, not both partOfPT5) beats it."""
    eta = ev['t5_eta'].astype(np.float32)
    phi = ev['t5_phi'].astype(np.float32)
    dnn = ev['t5_dnnScore'].astype(np.float32)
    emb = np.array([np.asarray(x) for x in ev['t5_embed']]) if len(ev['t5_embed']) else np.zeros((0, 6))
    cand = [i for i in idxs if not abdup[i]]
    ca = np.array(cand, int)
    dead = set()
    if not len(ca):
        return dead
    for i in cand:
        m = (np.abs(eta[ca] - eta[i]) <= 0.1) & (np.abs(dphi(phi[ca], phi[i])) <= 0.1) & (ca != i)
        for j in ca[m]:
            if part[i] and part[j]:
                continue
            n1 = count_in(t5hits[i], t5sets[j])
            n2 = count_in(t5hits[j], t5sets[i])
            d2 = float(np.sum((emb[i] - emb[j]) ** 2))
            isd = lambda n: (n >= 5 and d2 < 0.25) or n >= 10
            if isd(n1) or isd(n2):
                if dnn[i] < dnn[j] or (dnn[i] == dnn[j] and i < j):
                    dead.add(i)
                    break
    return dead


def cc_dead(ev, H, part, dup_any, t5hits, alive_pt5, pt5_ot, pt5_eta, pt5_phi, pt3_list):
    eta = ev['t5_eta'].astype(np.float32)
    phi = ev['t5_phi'].astype(np.float32)
    dead = set()
    for i in range(len(eta)):
        if dup_any[i] or part[i]:
            continue
        hs = set(t5hits[i])
        for (e, p, ot) in [(pt5_eta[k], pt5_phi[k], pt5_ot[k]) for k in alive_pt5] + pt3_list:
            if abs(eta[i] - e) >= 0.15 or abs(dphi(phi[i], p)) >= 0.15:
                continue
            if sum(1 for h in hs if h in ot) >= 4:
                dead.add(i)
                break
    return dead


def main():
    C = collections.Counter()
    for ie, ev in events():
        H = Hits(ev)
        n5 = len(ev['t5_eta'])
        bits = ev['t5_isDupBits'].astype(int)
        part = ev['t5_partOfPT5'].astype(bool)
        abdup = (bits & 1).astype(bool)
        t5hits = [H.t5_hits(t) for t in range(n5)]
        t5sets = [set(h) for h in t5hits]
        pdup = ev['pT5_isDupReco'].astype(bool)
        p_t5 = ev['pT5_t5Idx'].astype(int)
        alive_pt5 = np.nonzero(~pdup)[0]
        has_alive = np.zeros(n5, bool)
        has_alive[p_t5[~pdup]] = True
        part_new = part & has_alive
        pt5_ot = {k: set(t5hits[p_t5[k]]) for k in range(len(p_t5))}
        pt5_eta = f16(ev['t5_eta'][p_t5]) if len(p_t5) else []
        pt5_phi = f16(ev['t5_phi'][p_t5]) if len(p_t5) else []
        tc_type = ev['tc_type'].astype(int)
        pt3_list = []
        for k in np.nonzero(tc_type == 5)[0]:
            q = int(ev['tc_pt3Idx'][k])
            pt3_list.append((float(ev['pT3_eta'][q]), float(ev['pT3_phi'][q]), set(H.t3_hits(int(ev['pT3_t3Idx'][q])))))
        # ---- baseline validation
        b_btc = btc_dead(ev, H, part, abdup, t5hits, t5sets, range(n5))
        act_btc = set(np.nonzero(bits & 2)[0].tolist())
        C['btc_rep'] += len(b_btc); C['btc_act'] += len(act_btc); C['btc_agree'] += len(b_btc & act_btc)
        dup_b = abdup.copy()
        for i in b_btc: dup_b[i] = True
        b_cc = cc_dead(ev, H, part, abdup | ((bits & 2) > 0), t5hits, alive_pt5, pt5_ot, pt5_eta, pt5_phi, pt3_list)
        act_cc = set(np.nonzero(bits & 4)[0].tolist())
        C['cc_rep'] += len(b_cc); C['cc_act'] += len(act_cc); C['cc_agree'] += len(b_cc & act_cc)
        tc_t5_act = set(int(x) for x, t in zip(ev['tc_t5Idx'], tc_type) if t == 4)
        # ---- unstale
        stale = np.nonzero(part & ~has_alive)[0]
        C['stale_T5'] += len(stale)
        C['stale_T5_notAB'] += int((~abdup[stale]).sum())
        n_btc = btc_dead(ev, H, part_new, abdup, t5hits, t5sets, range(n5))
        dup_n = abdup | np.isin(np.arange(n5), list(n_btc))
        n_cc = cc_dead(ev, H, part_new, dup_n, t5hits, alive_pt5, pt5_ot, pt5_eta, pt5_phi, pt3_list)
        tc_t5_new = set(i for i in range(n5) if not dup_n[i] and not part_new[i] and i not in n_cc)
        # use actual baseline T5 TC set as reference; changes = replay(new) - replay(base) applied to actual
        tc_t5_base_rep = set(i for i in range(n5) if not abdup[i] and i not in b_btc and not part[i] and i not in b_cc)
        C['base_T5TC_rep'] += len(tc_t5_base_rep); C['base_T5TC_act'] += len(tc_t5_act)
        C['base_T5TC_agree'] += len(tc_t5_base_rep & tc_t5_act)
        added = tc_t5_new - tc_t5_base_rep
        removed = tc_t5_base_rep - tc_t5_new
        C['T5TC_added'] += len(added); C['T5TC_removed'] += len(removed)
        C['T5TC_added_fake'] += int(sum(ev['t5_isFake'][i] for i in added))
        C['T5TC_removed_fake'] += int(sum(ev['t5_isFake'][i] for i in removed))
        # T4 TCs killed by new T5 TCs (CrossCleanT4: >=3 shared hits)
        t4_lost = set()
        for k in np.nonzero(tc_type == 9)[0]:
            t4 = int(ev['tc_t4Idx'][k]); h4 = set(H.t4_hits(t4))
            if any(len(h4 & t5sets[i]) >= 3 for i in added):
                t4_lost.add(int(k))
        C['T4TC_lost'] += len(t4_lost)
        # sims
        for tag, drm in (('02', 0.02), ('10', 0.10)):
            den = denominator(ev, drm)
            for s in np.nonzero(den)[0]:
                base = ev['sim_tcIdx'][s] >= 0
                t5g = set(int(x) for x in genuine(ev, 't5', s))
                tcs = [int(k) for k, f in zip(ev['sim_tcIdxAll'][s], ev['sim_tcIdxAllFrac'][s]) if f > TCMATCH]
                still = [k for k in tcs if not (tc_type[k] == 4 and int(ev['tc_t5Idx'][k]) in removed)
                         and k not in t4_lost]
                new = bool(still) or bool(t5g & added)
                C['den' + tag] += 1
                C['gain' + tag] += new and not base
                C['loss' + tag] += base and not new
                if not base and (t5g & set(stale.tolist())):
                    C['fail_with_stale_genuine_T5_' + tag] += 1
        print(ie, dict(C), file=sys.stderr, flush=True)
    print(dict(C))
    for tag in ('02', '10'):
        net = C['gain' + tag] - C['loss' + tag]
        print(f"dR<{'0.02' if tag == '02' else '0.10'}: +{C['gain' + tag]} -{C['loss' + tag]} net {net:+d} "
              f"({100 * net / C['den' + tag]:+.2f} pp of {C['den' + tag]})")


if __name__ == '__main__':
    main()
