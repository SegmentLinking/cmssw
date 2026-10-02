#!/usr/bin/env python3
"""For jet-core failing tracks with truth-matched T5s (Exp F ntuple): how many of
their matched T5s were consumed into a pT5 object (partOfPT5=true)?
If consumed: what did that pT5 truth-match to, and at what fraction?
"""
import sys
import ROOT

FNAME = sys.argv[1] if len(sys.argv) > 1 else "LSTNtuple_noT5pT5dedup.root"
FRAC = 0.75

f = ROOT.TFile(FNAME)
t = f.Get("tree")

n_fail = 0
all_consumed = 0
some_free = 0
none_consumed = 0
pt5_match_info = []  # (matches_this_track_frac, best_other_frac)

for ev in t:
    dr = ev.sim_genjet_deltaR
    jidx = ev.sim_genjet_idx
    gpt = ev.genjet_pt
    geta = ev.genjet_eta
    pt5_t5 = list(ev.pT5_t5Idx)
    t5_to_pt5 = {}
    for ipt5, it5 in enumerate(pt5_t5):
        t5_to_pt5.setdefault(it5, []).append(ipt5)
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
        t5s = [
            idx for idx, fr in zip(ev.sim_t5IdxAll[i], ev.sim_t5IdxAllFrac[i]) if fr >= FRAC
        ]
        if not t5s:
            continue
        n_fail += 1
        consumed = [j for j in t5s if j in t5_to_pt5]
        free = [j for j in t5s if j not in t5_to_pt5]
        if not consumed:
            none_consumed += 1
        elif not free:
            all_consumed += 1
        else:
            some_free += 1
        for j in consumed:
            for ipt5 in t5_to_pt5[j]:
                sims = list(ev.pT5_simIdxAll[ipt5])
                fracs = list(ev.pT5_simIdxAllFrac[ipt5])
                this_frac = max(
                    (fr for s, fr in zip(sims, fracs) if s == i), default=0.0
                )
                other_frac = max(
                    (fr for s, fr in zip(sims, fracs) if s != i), default=0.0
                )
                pt5_match_info.append((this_frac, other_frac))

print(f"jet-core (dR<0.01) failing tracks with matched T5: {n_fail}")
print(f"  ALL matched T5s consumed by a pT5 (partOfPT5): {all_consumed}")
print(f"  some T5s free (not in any pT5):                {some_free}")
print(f"  no T5 consumed:                                {none_consumed}")
if pt5_match_info:
    print(f"\nconsuming pT5s ({len(pt5_match_info)}):")
    hi_this = sum(1 for a, b in pt5_match_info if a >= FRAC)
    hi_other = sum(1 for a, b in pt5_match_info if b >= FRAC and a < FRAC)
    lo_both = sum(1 for a, b in pt5_match_info if a < FRAC and b < FRAC)
    print(f"  pT5 matches THIS track >=0.75: {hi_this}")
    print(f"  pT5 matches OTHER track only:  {hi_other}")
    print(f"  pT5 matches nobody (fake):     {lo_both}")
    print("  (this_frac, other_frac) samples:",
          " ".join(f"({a:.2f},{b:.2f})" for a, b in pt5_match_info[:20]))
