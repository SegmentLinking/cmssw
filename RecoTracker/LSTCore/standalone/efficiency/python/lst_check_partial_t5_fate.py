#!/usr/bin/env python3
"""For the category-D failing jet-core tracks: the fate of ALL their partially
matched T5s (0 < frac < 0.75) — the potential dedup-cluster winners.
Fates: consumed by pT5 (partOfPT5), killed after-build (bit0), killed before-TC
(bit1), survivor (dupbits==0; partOfTC or not).
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
n_with_partial = 0
fate_counter = Counter()
per_track_fates = Counter()

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
            continue
        if not [j for j, fr in matched if not popt5[j]]:
            continue
        n_catD += 1
        partial = [(j, fr) for j, fr in allt5 if 0.0 < fr < FRAC]
        if not partial:
            per_track_fates[("NO_PARTIAL_T5S",)] += 1
            continue
        n_with_partial += 1
        fates = set()
        for j, fr in partial:
            if popt5[j]:
                fate = "consumed_by_pT5"
            elif dupbits[j] == 0:
                fate = "survivor_inTC" if ptc[j] else "survivor_noTC"
            elif dupbits[j] & 1:
                fate = "killed_afterBuild"
            else:
                fate = "killed_beforeTC"
            fate_counter[fate] += 1
            fates.add(fate)
        per_track_fates[tuple(sorted(fates))] += 1

print(f"category-D failing core tracks: {n_catD} ({n_with_partial} with partial T5s)")
print(f"\nper-T5 fates of partial (0<frac<0.75) T5s:")
for fate, n in fate_counter.most_common():
    print(f"  {fate:>20}: {n}")
print(f"\nper-track fate sets:")
for fs, n in sorted(per_track_fates.items(), key=lambda x: -x[1]):
    print(f"  {n:3d}  {list(fs)}")
