#!/usr/bin/env python3
"""For each category-D failing jet-core track: find ALL T5s within dR<0.01 of its
killed matched T5s (the dedup-cluster neighborhood) and tabulate their reco status
and purity (t5_pMatched, which is NOT 75%-thresholded).
The after-build dedup winner(s) must be among the survivors here.
Run on LSTNtuple_expF_diag.root.
"""
import math
import sys
from collections import Counter
import ROOT

FNAME = sys.argv[1] if len(sys.argv) > 1 else "LSTNtuple_expF_diag.root"
FRAC = 0.75
CONE = 0.01

f = ROOT.TFile(FNAME)
t = f.Get("tree")


def dphi(a, b):
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


n_catD = 0
n_with_surv = 0
n_without_surv = 0
surv_status = Counter()
surv_pmatched = []
best_surv_pm_per_track = []

for ev in t:
    dr = ev.sim_genjet_deltaR
    jidx = ev.sim_genjet_idx
    gpt = ev.genjet_pt
    geta = ev.genjet_eta
    dupbits = ev.t5_isDupBits
    popt5 = ev.t5_partOfPT5
    ptc = ev.t5_partOfTC
    pm = ev.t5_pMatched
    t5eta = ev.t5_eta
    t5phi = ev.t5_phi
    nt5 = len(t5eta)
    nsim = len(ev.sim_pt)
    for i in range(nsim):
        if ev.sim_q[i] != -1:
            continue
        if not (ev.sim_pt[i] > 0.9 and abs(ev.sim_eta[i]) < 4.5):
            continue
        if not (abs(ev.sim_vz[i]) < 30 and (ev.sim_vx[i] ** 2 + ev.sim_vy[i] ** 2) ** 0.5 < 2.5):
            continue
        ji = jidx[i]
        if ji < 0 or not (gpt[ji] > 1000 and abs(geta[ji]) < 2.5):
            continue
        if not (dr[i] < 0.01):
            continue
        if ev.sim_tcIdx[i] >= 0:
            continue
        matched = [
            idx for idx, fr in zip(ev.sim_t5IdxAll[i], ev.sim_t5IdxAllFrac[i]) if fr >= FRAC
        ]
        pt5s = [
            idx for idx, fr in zip(ev.sim_pt5IdxAll[i], ev.sim_pt5IdxAllFrac[i]) if fr >= FRAC
        ]
        if pt5s or not matched:
            continue
        if not [j for j in matched if not popt5[j]]:
            continue
        n_catD += 1
        # neighborhood = union of cones around killed matched T5s
        seeds = [(t5eta[j], t5phi[j]) for j in matched]
        matched_set = set(matched)
        surv = []
        for k in range(nt5):
            if k in matched_set:
                continue
            if dupbits[k] != 0:
                continue  # only survivors of dedup
            e, p = t5eta[k], t5phi[k]
            if any(
                math.sqrt((e - e0) ** 2 + dphi(p, p0) ** 2) < CONE for e0, p0 in seeds
            ):
                surv.append(k)
        if not surv:
            n_without_surv += 1
            continue
        n_with_surv += 1
        best_pm = 0.0
        for k in surv:
            status = []
            status.append("inPT5" if popt5[k] else ("inTC" if ptc[k] else "free"))
            pmk = pm[k]
            if pmk >= FRAC:
                pur = "pure(other trk)"
            elif pmk >= 0.5:
                pur = "mixed 0.5-0.75"
            elif pmk > 0.0:
                pur = "low <0.5"
            else:
                pur = "zero"
            surv_status[(status[0], pur)] += 1
            surv_pmatched.append(pmk)
            best_pm = max(best_pm, pmk)
        best_surv_pm_per_track.append(best_pm)

print(f"category-D failing core tracks: {n_catD}")
print(f"  with >=1 surviving (isDupBits==0) T5 within dR<{CONE} of killed cluster: {n_with_surv}")
print(f"  with NO survivor in cone:                                             {n_without_surv}")
print(f"\nsurviving-T5 (status, purity) counts [{sum(surv_status.values())} T5s]:")
for (st, pur), n in surv_status.most_common():
    print(f"  {st:>6} | {pur:>15}: {n}")
if best_surv_pm_per_track:
    v = sorted(best_surv_pm_per_track)
    n = len(v)
    print(f"\nbest survivor pMatched per track: median {v[n//2]:.3f}  "
          f"q25 {v[n//4]:.3f}  q75 {v[3*n//4]:.3f}")
    print("  values:", " ".join(f"{x:.2f}" for x in v))
