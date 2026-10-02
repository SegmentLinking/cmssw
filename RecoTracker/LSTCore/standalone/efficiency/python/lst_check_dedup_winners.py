#!/usr/bin/env python3
"""For category-D failing jet-core tracks (free matched T5 exists, no TC): inspect
the SURVIVING T5s (reco isDupBits==0) that are partially matched (0 < frac < 0.75)
to the same sim track — the dedup-cluster winners. Did the winner:
  - have sub-threshold purity (mixed hits)?
  - become part of a TC (a 'fake' TC that is really this track)?
Run on LSTNtuple_expF_diag.root.
"""
import sys
from collections import Counter
import ROOT

FNAME = sys.argv[1] if len(sys.argv) > 1 else "LSTNtuple_expF_diag.root"
FRAC = 0.75

f = ROOT.TFile(FNAME)
t = f.Get("tree")

n_catD = 0
have_surv_partial = 0
no_surv_partial = 0
winner_fracs = []
winner_tc = Counter()  # partOfTC of best surviving partial winner
best_frac_per_track = []

for ev in t:
    dr = ev.sim_genjet_deltaR
    jidx = ev.sim_genjet_idx
    gpt = ev.genjet_pt
    geta = ev.genjet_eta
    dupbits = ev.t5_isDupBits
    popt5 = ev.t5_partOfPT5
    ptc = ev.t5_partOfTC
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
        allt5 = list(zip(ev.sim_t5IdxAll[i], ev.sim_t5IdxAllFrac[i]))
        matched = [(j, fr) for j, fr in allt5 if fr >= FRAC]
        pt5s = [
            idx for idx, fr in zip(ev.sim_pt5IdxAll[i], ev.sim_pt5IdxAllFrac[i]) if fr >= FRAC
        ]
        if pt5s or not matched:
            continue  # not category D
        free = [j for j, fr in matched if not popt5[j]]
        if not free:
            continue
        n_catD += 1
        # surviving partially-matched T5s
        surv = [
            (j, fr)
            for j, fr in allt5
            if fr < FRAC and dupbits[j] == 0
        ]
        if surv:
            have_surv_partial += 1
            best = max(surv, key=lambda x: x[1])
            best_frac_per_track.append(best[1])
            winner_fracs.extend(fr for _, fr in surv)
            winner_tc[bool(ptc[best[0]])] += 1
        else:
            no_surv_partial += 1

print(f"category-D failing core tracks: {n_catD}")
print(f"  with a SURVIVING partial-match T5 (winner, frac<0.75): {have_surv_partial}")
print(f"  with NO surviving partial T5 (winner unrelated):       {no_surv_partial}")
if best_frac_per_track:
    v = sorted(best_frac_per_track)
    n = len(v)
    print(f"\nbest surviving-winner frac per track: median {v[n//2]:.3f}  "
          f"q25 {v[n//4]:.3f}  q75 {v[3*n//4]:.3f}")
    print("  values:", " ".join(f"{x:.2f}" for x in v))
    print(f"\nbest winner partOfTC: {dict(winner_tc)}")
