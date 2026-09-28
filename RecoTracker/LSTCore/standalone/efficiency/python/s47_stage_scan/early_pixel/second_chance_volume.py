#!/usr/bin/env python3
"""Volume of pass1-flagged pLS whose every killer produced no pT5/pT3/pLS-TC (second-chance candidates), 100 evt."""
import os, sys, collections, numpy as np, uproot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pls_common import *
BASE = "/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/"
L = uproot.open(BASE + "Ntuple-files/LSTNtuple_s45_rebased_100evt.root:tree").arrays(
    ["sim_pt", "pT5_plsIdx", "pT3_plsIdx", "tc_plsIdx", "pLS_isFake", "pLS_simIdxAll", "pLS_simIdxAllFrac", "sim_tcIdx"], library="np")
ib = ["sim_pt", "sim_bunchCrossing", "sim_event", "see_algo", "see_hitIdx", "see_hitType", "see_stateTrajGlbPx",
      "see_stateTrajGlbPy", "see_stateTrajGlbPz", "see_stateTrajGlbX", "see_stateTrajGlbY", "see_stateTrajGlbZ",
      "see_px", "see_py", "see_pz", "see_dxy", "see_dz", "see_ptErr", "pix_simHitIdx", "ph2_simHitIdx", "simhit_simTrkIdx"]
I = uproot.open(BASE + "trackingNtuple-100.root:trackingNtuple/tree").arrays(ib, library="np")
l2i = event_map(L["sim_pt"], I["sim_pt"], I["sim_bunchCrossing"], I["sim_event"])
G = collections.Counter()
for le in range(100):
    seeds = build_seeds({k: I[k][l2i[le]] for k in ib})
    li = [s for s, sd in enumerate(seeds) if sd["lst"]]
    ph = [seeds[s]["ph"] for s in li]; quad = np.array([seeds[s]["quad"] for s in li])
    score = np.array([seeds[s]["score"] for s in li], dtype=np.float32)
    d, k = checkhits_pass1(ph, quad, score, pair_list([seeds[s]["eta"] for s in li]), False)
    used = set(L["pT5_plsIdx"][le].tolist()) | set(L["pT3_plsIdx"][le].tolist()) | set(x for x in L["tc_plsIdx"][le].tolist() if x >= 0)
    fake = L["pLS_isFake"][le]; stc = L["sim_tcIdx"][le]
    for p in np.nonzero(d)[0]:
        G["flagged"] += 1
        if all(w not in used for (w, _, _) in k[p]):
            G["second_chance"] += 1
            G["second_chance_fake"] += int(fake[p])
            sims = [s for s, f in zip(L["pLS_simIdxAll"][le][p], L["pLS_simIdxAllFrac"][le][p]) if f > 0.75]
            G["second_chance_sim_noTC"] += int(bool(sims) and not any(stc[s] >= 0 for s in sims))
    G["alive"] += int((~d).sum())
print(dict(G))
