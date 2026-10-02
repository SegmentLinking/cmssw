#!/usr/bin/env python3
"""Master classification of jet-core (dR<0.01) failing denominator tracks in Exp F ntuple.

Categories (first match wins):
  A. no matched T5 and no matched pT5      -> T5-building reservoir gap
  B. matched pT5 exists (>=0.75), no TC    -> pT5-vs-pT5 dedup victim
  C. matched T5, all consumed by pT5s      -> partOfPT5 blocker
  D. matched T5 with a free (unconsumed) T5-> T5 dedup stages / tightCutFlag
Also runs on far region (dR>0.05) for comparison.
"""
import sys
import ROOT

FNAME = sys.argv[1] if len(sys.argv) > 1 else "LSTNtuple_noT5pT5dedup.root"
FRAC = 0.75

f = ROOT.TFile(FNAME)
t = f.Get("tree")

res = {}

for ev in t:
    dr = ev.sim_genjet_deltaR
    jidx = ev.sim_genjet_idx
    gpt = ev.genjet_pt
    geta = ev.genjet_eta
    consumed_t5 = set(ev.pT5_t5Idx)
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
        if dr[i] < 0.01:
            region = "core"
        elif dr[i] > 0.05:
            region = "far"
        else:
            continue
        r = res.setdefault(region, {"denom": 0, "pass": 0, "A": 0, "B": 0, "C": 0, "D": 0})
        r["denom"] += 1
        if ev.sim_tcIdx[i] >= 0:
            r["pass"] += 1
            continue
        t5s = [
            idx for idx, fr in zip(ev.sim_t5IdxAll[i], ev.sim_t5IdxAllFrac[i]) if fr >= FRAC
        ]
        pt5s = [
            idx for idx, fr in zip(ev.sim_pt5IdxAll[i], ev.sim_pt5IdxAllFrac[i]) if fr >= FRAC
        ]
        if pt5s:
            r["B"] += 1
        elif not t5s:
            r["A"] += 1
        elif all(j in consumed_t5 for j in t5s):
            r["C"] += 1
        else:
            r["D"] += 1

for region in ("core", "far"):
    r = res.get(region)
    if not r:
        continue
    nf = r["denom"] - r["pass"]
    print(f"[{region}] denom {r['denom']}  pass {r['pass']} "
          f"({r['pass']/r['denom']:.4f})  failing {nf}")
    print(f"   A no T5, no pT5 (build gap):        {r['A']}")
    print(f"   B matched pT5 exists, no TC:        {r['B']}   <- pT5 dedup victim")
    print(f"   C all T5s consumed by pT5:          {r['C']}")
    print(f"   D free T5 exists, no TC:            {r['D']}   <- T5 dedup / tightCut")
