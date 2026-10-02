#!/usr/bin/env python3
"""Rule (b): is there an LST-available seed quantity that ranks the genuine pLS above its contaminated
same-sim duplicate in CheckHitspLS pass 1? Global over 100 evt, all pairs that trigger the >=3 test where exactly one
of the two is genuine (fr>0.75) for a sim that the other also carries (fr>=0.5). Quantities: score_lsq (current),
ptErr/pt, etaErr, see_chi2 (not in LST input; for reference)."""
import os, sys, collections, numpy as np, uproot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pls_common import *
BASE = "/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/"
ib = ["sim_pt", "sim_bunchCrossing", "sim_event", "see_algo", "see_hitIdx", "see_hitType", "see_stateTrajGlbPx",
      "see_stateTrajGlbPy", "see_stateTrajGlbPz", "see_stateTrajGlbX", "see_stateTrajGlbY", "see_stateTrajGlbZ",
      "see_px", "see_py", "see_pz", "see_dxy", "see_dz", "see_ptErr", "see_etaErr", "see_chi2", "pix_simHitIdx",
      "ph2_simHitIdx", "simhit_simTrkIdx"]
I = uproot.open(BASE + "trackingNtuple-100.root:trackingNtuple/tree").arrays(ib, library="np")
stat = collections.Counter()
for ie in range(100):
    E = {k: I[k][ie] for k in ib}
    seeds = build_seeds(E)
    li = [s for s, sd in enumerate(seeds) if sd["lst"]]
    ph = [seeds[s]["ph"] for s in li]; quad = np.array([seeds[s]["quad"] for s in li])
    score = np.array([seeds[s]["score"] for s in li], dtype=np.float32)
    ptrel = np.array([seeds[s]["ptErr"] / seeds[s]["ptIn"] for s in li])
    etaerr = np.array([E["see_etaErr"][s] for s in li]); chi2 = np.array([E["see_chi2"][s] for s in li])
    for i, j in pair_list([seeds[s]["eta"] for s in li]):
        npm = sum(1 for h in ph[i] if h in ph[j])
        if npm < 3 or quad[i] != quad[j]:
            continue
        fi, fj = seeds[li[i]]["fr"], seeds[li[j]]["fr"]
        gi = [t for t, f in fi.items() if f > 0.75 and fj.get(t, 0) >= 0.5]
        gj = [t for t, f in fj.items() if f > 0.75 and fi.get(t, 0) >= 0.5]
        if bool(gi) == bool(gj):
            continue
        g, b = (i, j) if gi else (j, i)
        typ = ("quad" if quad[i] else "trip") + f"_nd{len(set(ph[i]) & set(ph[j]))}"
        stat[(typ, "n")] += 1
        for name, v in [("score", score), ("ptErr/pt", ptrel), ("etaErr", etaerr), ("chi2", chi2)]:
            stat[(typ, name)] += int(v[g] < v[b])
for typ in sorted(set(k[0] for k in stat)):
    n = stat[(typ, "n")]
    print(f"{typ:10s} n={n:5d}  genuine ranked first by: " + "  ".join(f"{nm}={stat[(typ, nm)] / n:.2f}" for nm in ["score", "ptErr/pt", "etaErr", "chi2"]))
