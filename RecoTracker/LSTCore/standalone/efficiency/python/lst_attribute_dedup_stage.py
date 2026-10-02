#!/usr/bin/env python3
"""Gate attribution for jet-core TC failures using the reco-flag branches
(t5_isDupBits, t5_tightCutFlag, t5_partOfPT5, pT5_isDupReco) added in Session 21.

Run on LSTNtuple_expF_diag.root (F1+F2 state, no D).

For each failing denominator track (dR<0.01, q=-1):
  - category B (matched pT5, no TC): confirm pT5_isDupReco==1 on all matched pT5s
  - category D (free matched T5, no TC): per free T5, report the blocking gate:
      isDupBits bit0 (=1): after-build dedup (or residual CrossCleanT5 vs-pT3)
      isDupBits bit1 (=2): before-TC standalone-vs-standalone dedup
      isDupBits==0 & !tightCutFlag: tightCut gate
      isDupBits==0 & tightCutFlag & !partOfPT5: should have been a TC (inconsistency!)
Track-level attribution: a track is "rescuable by stage X" if ALL its free T5s are
blocked only by stage X; mixed otherwise.
"""
import sys
from collections import Counter
import ROOT

FNAME = sys.argv[1] if len(sys.argv) > 1 else "LSTNtuple_expF_diag.root"
FRAC = 0.75

f = ROOT.TFile(FNAME)
t = f.Get("tree")

catB_dup = 0
catB_notdup = 0
t5_gate_counter = Counter()   # per-T5 gate
trk_gate_counter = Counter()  # per-track dominant gate set
n_core_denom = 0
n_core_pass = 0

for ev in t:
    dr = ev.sim_genjet_deltaR
    jidx = ev.sim_genjet_idx
    gpt = ev.genjet_pt
    geta = ev.genjet_eta
    dupbits = ev.t5_isDupBits
    tight = ev.t5_tightCutFlag
    popt5 = ev.t5_partOfPT5
    pt5dup = ev.pT5_isDupReco
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
        n_core_denom += 1
        if ev.sim_tcIdx[i] >= 0:
            n_core_pass += 1
            continue
        t5s = [
            idx for idx, fr in zip(ev.sim_t5IdxAll[i], ev.sim_t5IdxAllFrac[i]) if fr >= FRAC
        ]
        pt5s = [
            idx for idx, fr in zip(ev.sim_pt5IdxAll[i], ev.sim_pt5IdxAllFrac[i]) if fr >= FRAC
        ]
        if pt5s:
            if all(pt5dup[j] for j in pt5s):
                catB_dup += 1
            else:
                catB_notdup += 1
            continue
        if not t5s:
            continue  # category A
        free = [j for j in t5s if not popt5[j]]
        if not free:
            continue  # category C
        gates = set()
        for j in free:
            b = dupbits[j]
            if b & 2:
                g = "beforeTC_dedup" if not (b & 1) else "both_dedups"
            elif b & 1:
                g = "afterBuild_dedup"
            elif not tight[j]:
                g = "tightCut"
            else:
                g = "NONE(should be TC!)"
            t5_gate_counter[g] += 1
            gates.add(g)
        trk_gate_counter[frozenset(gates)] += 1

print(f"core denom {n_core_denom}  pass {n_core_pass}  failing {n_core_denom-n_core_pass}")
print(f"\ncategory B (matched pT5, no TC):")
print(f"  all matched pT5s have reco isDup=1: {catB_dup}")
print(f"  some matched pT5 NOT dup (?!):      {catB_notdup}")
print(f"\ncategory D per-T5 gate attribution ({sum(t5_gate_counter.values())} free T5s):")
for g, n in t5_gate_counter.most_common():
    print(f"  {g:>22}: {n}")
print(f"\ncategory D per-track gate sets:")
for gs, n in sorted(trk_gate_counter.items(), key=lambda x: -x[1]):
    print(f"  {n:3d}  {sorted(gs)}")
